# ignitefp4_lib/quantization_utils.py
"""Utilities for preparing and converting PyTorch models for FP4 quantization.

This module provides functions for both Post-Training Quantization (PTQ) and
Quantization-Aware Training (QAT) workflows within the IgniteFP4 framework.
It includes tools to:
- Recursively traverse a PyTorch model.
- Replace standard `torch.nn` layers (e.g., Linear, Conv2d, BatchNorm2d)
  with their FP4-simulating counterparts from `ignitefp4_lib.layers`.
- Insert `FP4QuantStub` and `FP4DequantStub` modules at appropriate locations
  to manage the quantization and dequantization of activations.
- Calibrate quantization parameters (scale and zero-point) for activations
  using a representative dataset (for PTQ).
- Convert models to an evaluation-ready state after PTQ or QAT.

The core idea is to modify a floating-point model by swapping out standard
layers with FP4-aware layers and adding stubs to handle activation quantization.
The `FP4QuantStub` modules are central to this, operating in different modes
for PTQ (observing statistics) and QAT (applying fake quantization during training).
"""
import torch
import torch.nn as nn
from .layers import FP4Linear, FP4Conv2d, FP4BatchNorm2d, FP4QuantStub, FP4DequantStub, FP4AvgPool2d, FP4MaxPool2d
import copy

def _replace_module_with_fp4_version(module_fp32, module_name_str, model_fp4_parent, fp4_quant_stub_class=FP4QuantStub):
    """
    Helper to replace a single nn.Module with its FP4 equivalent and add a QuantStub.
    Returns the new FP4 module (or sequence including stub).
    This is a simplified helper. A full solution needs to handle module hierarchy.

    Note: This function is currently not used by the `_recursive_prepare` method,
    which implements a more direct in-place replacement logic. It's kept for
    potential future use or alternative refactoring.

    Args:
        module_fp32 (nn.Module): The original float32 module to be replaced.
        module_name_str (str): The name of the module within its parent.
        model_fp4_parent (nn.Module): The parent module where `module_fp32` resides.
        fp4_quant_stub_class (type): The class to use for quantization stubs,
                                     defaults to `FP4QuantStub`.

    Returns:
        nn.Sequential or nn.Module or None:
        A `nn.Sequential` containing an `FP4QuantStub` and the new FP4 layer if
        the module type is supported (Linear, Conv2d, BatchNorm2d).
        An FP4 pooling layer if the module is a supported pooling type.
        Returns `None` if the module type is not directly replaced by this helper.
    """
    fp4_module = None
    stub = fp4_quant_stub_class(qat_mode=False) # PTQ mode

    if isinstance(module_fp32, nn.Linear):
        fp4_module = FP4Linear(
            module_fp32.in_features,
            module_fp32.out_features,
            bias=module_fp32.bias is not None,
            device=module_fp32.weight.device,
            dtype=module_fp32.weight.dtype
        )
        fp4_module.load_from_float_linear(module_fp32)
        # Replace in parent:
        # setattr(model_fp4_parent, module_name_str, nn.Sequential(stub, fp4_module))
        return nn.Sequential(stub, fp4_module) # Return sequence to be set by caller

    elif isinstance(module_fp32, nn.Conv2d):
        fp4_module = FP4Conv2d(
            module_fp32.in_channels,
            module_fp32.out_channels,
            module_fp32.kernel_size,
            stride=module_fp32.stride,
            padding=module_fp32.padding,
            dilation=module_fp32.dilation,
            groups=module_fp32.groups,
            bias=module_fp32.bias is not None,
            padding_mode=module_fp32.padding_mode,
            device=module_fp32.weight.device,
            dtype=module_fp32.weight.dtype
        )
        fp4_module.load_from_float_conv2d(module_fp32)
        return nn.Sequential(stub, fp4_module)

    elif isinstance(module_fp32, nn.BatchNorm2d):
        fp4_module = FP4BatchNorm2d(
            module_fp32.num_features,
            eps=module_fp32.eps,
            momentum=module_fp32.momentum,
            affine=module_fp32.affine,
            track_running_stats=module_fp32.track_running_stats,
            device=module_fp32.weight.device if module_fp32.affine else (module_fp32.running_mean.device if module_fp32.track_running_stats else None),
            # dtype needs careful handling if not all params exist
        )
        # Ensure dtype is passed if possible, might need a helper to get a valid dtype from the float module
        if hasattr(module_fp32, 'weight') and module_fp32.weight is not None:
             fp4_module = FP4BatchNorm2d( # Re-init with dtype if weight exists
                module_fp32.num_features, eps=module_fp32.eps, momentum=module_fp32.momentum,
                affine=module_fp32.affine, track_running_stats=module_fp32.track_running_stats,
                device=module_fp32.weight.device, dtype=module_fp32.weight.dtype
             )
        fp4_module.load_from_float_batchnorm2d(module_fp32)
        # BatchNorm often has activations before it, so stub might be before it
        return nn.Sequential(stub, fp4_module)

    elif isinstance(module_fp32, nn.AvgPool2d):
        return FP4AvgPool2d(
            kernel_size=module_fp32.kernel_size,
            stride=module_fp32.stride,
            padding=module_fp32.padding,
            ceil_mode=module_fp32.ceil_mode,
            count_include_pad=module_fp32.count_include_pad,
            divisor_override=module_fp32.divisor_override
        )
    elif isinstance(module_fp32, nn.MaxPool2d):
        return FP4MaxPool2d(
            kernel_size=module_fp32.kernel_size,
            stride=module_fp32.stride,
            padding=module_fp32.padding,
            dilation=module_fp32.dilation,
            return_indices=module_fp32.return_indices,
            ceil_mode=module_fp32.ceil_mode
        )
    elif isinstance(module_fp32, (nn.ReLU, nn.ReLU6)): # Example: quantize output of ReLU
        # Could also choose to quantize input to ReLU if it's standalone
        # For now, let's assume quantizing input to major FP4 layers is primary
        # return nn.Sequential(copy.deepcopy(module_fp32), stub)
        return copy.deepcopy(module_fp32) # Keep RelU as is for now

    # Add other modules as needed, e.g. nn.Identity, nn.Dropout etc.
    # For containers like nn.Sequential, we need to iterate and rebuild.

    return None # Indicate no replacement was made for this type directly by this helper


def prepare_model_for_ptq(model_fp32: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Prepares a float32 PyTorch model for Post-Training Quantization (PTQ) with IgniteFP4.

    This function traverses the input model and performs the following modifications:
    1.  Replaces supported `torch.nn` layers (e.g., `nn.Linear`, `nn.Conv2d`,
        `nn.BatchNorm2d`) with their corresponding FP4-simulating counterparts
        from `ignitefp4_lib.layers` (e.g., `FP4Linear`, `FP4Conv2d`, `FP4BatchNorm2d`).
        The weights and biases from the original float modules are loaded into these
        new FP4 modules.
    2.  Inserts `FP4QuantStub` modules immediately before each of these replaced FP4
        layers. These stubs are configured for PTQ (i.e., `qat_mode=False` and
        `learnable_params=False`), meaning they will observe activation statistics
        during a calibration phase.
    3.  Replaces standard pooling layers (`nn.AvgPool2d`, `nn.MaxPool2d`) with their
        simple FP4 wrapper counterparts (`FP4AvgPool2d`, `FP4MaxPool2d`), which
        pass through data without modification but maintain the FP4 layer type.
    4.  Other module types (e.g., `nn.ReLU`, `nn.Dropout`, `nn.Sequential` containers)
        are traversed, but their structures are generally preserved unless they contain
        children that are modified as per the rules above.

    The goal is to create a model that, after calibration, can simulate FP4 inference.

    Args:
        model_fp32 (nn.Module): The original float32 PyTorch model.
        inplace (bool): If `True`, the `model_fp32` is modified directly.
                        If `False` (default), a deep copy of `model_fp32` is
                        created and modified, leaving the original model unchanged.
                        Note: True in-place modification for arbitrary model structures
                        can be complex; current implementation defaults to deepcopy
                        if issues are anticipated, with a warning.

    Returns:
        nn.Module: The model prepared for FP4 PTQ. It contains FP4 layers and
                   `FP4QuantStub`s ready for calibration.
    """
    if not inplace:
        model_fp4 = copy.deepcopy(model_fp32)
    else:
        # In-place modification is harder to manage correctly for all model types.
        # For now, let's focus on the deepcopy approach for safety.
        # If inplace is True, we'd be modifying model_fp32 directly.
        print("Warning: `inplace=True` is not fully robust yet. Using deepcopy.")
        model_fp4 = copy.deepcopy(model_fp32)


    _recursive_prepare(model_fp4, "", qat_mode=False, learnable_params=False)
    return model_fp4

def _recursive_prepare(module: nn.Module, prefix: str, qat_mode: bool = False, learnable_params: bool = False):
    """
    Recursively traverses a PyTorch module, modifying it in-place for FP4 quantization.

    This is the core helper function used by `prepare_model_for_ptq` and
    `prepare_model_for_qat`. It iterates through all named children of the
    given `module`.

    For supported layer types (`nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`):
    - An `FP4QuantStub` is instantiated (with `qat_mode` and `learnable_params`
      forwarded).
    - The original layer is replaced by its FP4 counterpart (e.g., `FP4Linear`).
      Weights and parameters are copied from the original layer.
    - The original layer in the parent `module` is replaced by an `nn.Sequential`
      containing the new `FP4QuantStub` followed by the new FP4 layer.

    For supported pooling layers (`nn.AvgPool2d`, `nn.MaxPool2d`):
    - The original layer is replaced by its FP4 wrapper counterpart (`FP4AvgPool2d`,
      `FP4MaxPool2d`). No `FP4QuantStub` is typically inserted for pooling layers
      by default, as quantization usually happens before weight-bearing layers
      or non-linearities.

    For other module types:
    - If the module has children (e.g., `nn.Sequential`, custom `nn.Module` blocks),
      this function recursively calls itself on those children.
    - If the module is a leaf node and not one of the directly supported types for
      replacement (e.g., `nn.ReLU`, `nn.Dropout`), it is left unchanged.

    The modifications are performed by using `setattr(module, name, new_child_module)`
    on the parent `module`, effectively changing its structure.

    Args:
        module (nn.Module): The PyTorch module to be processed. Modifications
                            are applied in-place to this module's children.
        prefix (str): A string prefix representing the current module's hierarchy,
                      used for debugging or logging (e.g., "block1.conv1").
        qat_mode (bool): If `True`, `FP4QuantStub`s are configured for
                         Quantization-Aware Training (fake quantization). If `False`,
                         they are configured for Post-Training Quantization (statistic
                         observation). Defaults to `False`.
        learnable_params (bool): If `True` (and `qat_mode` is also `True`),
                                 `FP4QuantStub`s are configured with learnable
                                 quantization parameters (scale and zero-point).
                                 Defaults to `False`.
    """
    for name, child_module in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        # print(f"Processing module: {full_name} of type {type(child_module)}")

        # Attempt to replace with FP4 version + QuantStub
        # This is tricky because setattr changes the module, invalidating iterator if not careful.
        # We need to check type and then potentially replace.

        if isinstance(child_module, nn.Linear):
            stub = FP4QuantStub(qat_mode=qat_mode, learnable_params=learnable_params)
            fp4_layer = FP4Linear(
                child_module.in_features, child_module.out_features,
                bias=child_module.bias is not None,
                device=child_module.weight.device, dtype=child_module.weight.dtype
            )
            fp4_layer.load_from_float_linear(child_module)
            setattr(module, name, nn.Sequential(stub, fp4_layer))
            # print(f"  Replaced {full_name} with FP4Linear + Stub")

        elif isinstance(child_module, nn.Conv2d):
            stub = FP4QuantStub(qat_mode=qat_mode, learnable_params=learnable_params)
            fp4_layer = FP4Conv2d(
                child_module.in_channels, child_module.out_channels, child_module.kernel_size,
                stride=child_module.stride, padding=child_module.padding, dilation=child_module.dilation,
                groups=child_module.groups, bias=child_module.bias is not None,
                padding_mode=child_module.padding_mode, device=child_module.weight.device,
                dtype=child_module.weight.dtype
            )
            fp4_layer.load_from_float_conv2d(child_module)
            setattr(module, name, nn.Sequential(stub, fp4_layer))
            # print(f"  Replaced {full_name} with FP4Conv2d + Stub")

        elif isinstance(child_module, nn.BatchNorm2d):
            stub = FP4QuantStub(qat_mode=qat_mode, learnable_params=learnable_params) # Quantize input to BN
            fp4_layer = FP4BatchNorm2d(
                child_module.num_features, eps=child_module.eps, momentum=child_module.momentum,
                affine=child_module.affine, track_running_stats=child_module.track_running_stats
                # device/dtype will be handled by load_from_float
            )
            # Correctly get device/dtype for init before load
            _device = None
            _dtype = None
            if child_module.affine and child_module.weight is not None:
                _device = child_module.weight.device
                _dtype = child_module.weight.dtype
            elif child_module.track_running_stats and child_module.running_mean is not None:
                 _device = child_module.running_mean.device
                 _dtype = child_module.running_mean.dtype

            if _device is not None : #Re-init with device/dtype if found
                 fp4_layer = FP4BatchNorm2d(
                    child_module.num_features, eps=child_module.eps, momentum=child_module.momentum,
                    affine=child_module.affine, track_running_stats=child_module.track_running_stats,
                    device=_device, dtype=_dtype
                )
            fp4_layer.load_from_float_batchnorm2d(child_module)
            setattr(module, name, nn.Sequential(stub, fp4_layer))
            # print(f"  Replaced {full_name} with FP4BatchNorm2d + Stub")

        elif isinstance(child_module, nn.AvgPool2d):
            fp4_pool = FP4AvgPool2d(
                kernel_size=child_module.kernel_size, stride=child_module.stride, padding=child_module.padding,
                ceil_mode=child_module.ceil_mode, count_include_pad=child_module.count_include_pad,
                divisor_override=child_module.divisor_override
            )
            setattr(module, name, fp4_pool)
            # print(f"  Replaced {full_name} with FP4AvgPool2d")
             # Recursively process children of the new pooling layer if it were a container (it's not)

        elif isinstance(child_module, nn.MaxPool2d):
            fp4_pool = FP4MaxPool2d(
                kernel_size=child_module.kernel_size, stride=child_module.stride, padding=child_module.padding,
                dilation=child_module.dilation, return_indices=child_module.return_indices,
                ceil_mode=child_module.ceil_mode
            )
            setattr(module, name, fp4_pool)
            # print(f"  Replaced {full_name} with FP4MaxPool2d")

        # For nn.Sequential, we want to process its children.
        # If the child_module itself has children (e.g. it's a Sequential or a custom block), recurse.
        # The check `list(child_module.children())` is a common way to see if it has children.
        # However, our replacement with nn.Sequential(stub, fp4_layer) means the new module at `name`
        # is now a Sequential. We should recurse into the *original* child_module structure if it was a container.
        # This recursive logic needs to be careful. If we replace a module, we should then
        # recurse into the *new* module if it's a container of further modules we might want to process.
        # OR, more simply, if a module is NOT one of the ones we replace directly, we just recurse into it.

        else: # If not directly replaced, recurse into it if it has children
            # Get the potentially newly set module after a replacement above.
            # If child_module was replaced by nn.Sequential(stub, actual_fp4_layer),
            # we don't want to recurse into the stub or the fp4_layer directly here,
            # as their children are not meant to be further quantized by this generic recursion.
            # The FP4 layers themselves are now "black boxes" for this level of recursion.
            # The nn.Sequential wrapper itself might have children if it was user-defined and complex,
            # but the ones we create are simple.
            # So, we recurse only if the original child_module was a container and was *not* replaced.
            current_child_after_potential_setattr = getattr(module, name)
            if len(list(current_child_after_potential_setattr.children())) > 0 and \
               current_child_after_potential_setattr is child_module: # Recurse only if not replaced by our direct FP4 version
                # print(f"  Recursing into {full_name}...")
                _recursive_prepare(current_child_after_potential_setattr, full_name, qat_mode, learnable_params)
            elif isinstance(current_child_after_potential_setattr, nn.Sequential) and \
                 current_child_after_potential_setattr is not child_module:
                # This is a Sequential we just created (e.g. Stub + FP4Layer)
                # We might want to recurse into its children if they could be containers
                # but for Stub + FP4Layer, the FP4Layer is the "end" of this type of processing.
                # If the FP4Layer itself was a complex custom module, that'd be different.
                # For now, assume FP4Linear, FP4Conv2d etc. don't need further internal preparation here.
                # print(f"  Not recursing into the new Sequential wrapper for {full_name}")
                pass

def calibrate_model(model_fp4: nn.Module, calibration_dataloader: torch.utils.data.DataLoader, device: str = 'cpu'):
    """Calibrates a model prepared for PTQ by running calibration data through it and calculating quantization parameters.
    
    This function performs the crucial calibration step in the Post-Training Quantization (PTQ) workflow:
    
    1. Sets the model to training mode (`.train()`) - This is necessary to enable the `FP4QuantStub` modules 
       to collect activation statistics, even though no actual training happens.
    2. Processes batches from the `calibration_dataloader` through the model.
    3. During this process, each `FP4QuantStub` in the model observes and tracks the min/max values 
       of activations that pass through it.
    4. After all calibration data has been processed, calls `compute_quant_params()` on each `FP4QuantStub`
       to calculate its appropriate scale and zero-point based on the observed statistics.
    
    No gradients are computed during this process, and model weights remain unchanged. Only the 
    quantization parameters (scale/zero-point) in the `FP4QuantStub` modules are updated.
    
    Args:
        model_fp4 (nn.Module): The model prepared for PTQ using `prepare_model_for_ptq()`.
        calibration_dataloader (torch.utils.data.DataLoader): A DataLoader containing a small, 
                                                             representative dataset used to determine
                                                             activation ranges for quantization.
        device (str): The device to run calibration on ('cpu', 'cuda', etc.). Default: 'cpu'.
    
    Returns:
        None: The model is modified in-place, with `FP4QuantStub` modules being calibrated.
    
    Example:
        >>> # Prepare model
        >>> ptq_model = prepare_model_for_ptq(float_model)
        >>> # Create a small representative dataset
        >>> calibration_dataset = MyDataset(...)
        >>> calibration_loader = DataLoader(calibration_dataset, batch_size=32)
        >>> # Calibrate the model
        >>> calibrate_model(ptq_model, calibration_loader, device='cuda')
    """
    # Move model to specified device
    model_fp4.to(device)
    
    # Set model to training mode so QuantStubs observe statistics
    # This is counter-intuitive, but necessary since FP4QuantStub observes
    # activation statistics only in training mode
    model_fp4.train()
    
    # Disable gradient computation during calibration
    with torch.no_grad():
        # Process batches from the calibration dataset
        for batch_idx, data in enumerate(calibration_dataloader):
            # Handle different types of data batches
            if isinstance(data, (list, tuple)):
                # If data is a tuple/list (inputs, targets), take just the inputs
                inputs = data[0].to(device)
            else:
                # Otherwise assume data is just the inputs
                inputs = data.to(device)
            
            # Forward pass to collect activation statistics
            # No backward pass or weight updates occur
            _ = model_fp4(inputs)
    
    # Calculate quantization parameters for each FP4QuantStub based on observed statistics
    for name, module in model_fp4.named_modules():
        if isinstance(module, FP4QuantStub):
            # This computes scale and zero-point from the observed min/max values
            # and stores them in the module's buffers
            module.compute_quant_params()
            
            # Optional: Log the computed parameters
            # print(f"Calibrated {name}: scale={module.activation_scale.item():.6f}, "
            #       f"zero_point={module.activation_zp.item()}")

def convert_ptq_model_to_eval(model_fp4_calibrated: nn.Module) -> nn.Module:
    """Converts a calibrated PTQ model to evaluation mode for FP4-simulated inference.
    
    After a model has been calibrated with `calibrate_model()`, this function prepares it 
    for simulated FP4 inference by:
    
    1. Setting the model to evaluation mode (`.eval()`). This changes the behavior of 
       `FP4QuantStub` modules to perform actual quantization using their calibrated parameters
       rather than just observing statistics.
    2. Makes no structural changes to the model; the conversion is purely behavioral.
    
    In evaluation mode, all `FP4QuantStub` modules will quantize and dequantize activations
    using their calibrated scale and zero-point, simulating the effect of FP4 precision.
    The FP4 layer weights are already quantized when the model was prepared.
    
    Args:
        model_fp4_calibrated (nn.Module): A model that has been prepared for PTQ and calibrated.
        
    Returns:
        nn.Module: The same model, now in evaluation mode ready for FP4-simulated inference.
        
    Example:
        >>> # After calibration
        >>> eval_model = convert_ptq_model_to_eval(ptq_model)
        >>> # Now ready for FP4-simulated inference
        >>> with torch.no_grad():
        >>>     outputs = eval_model(test_inputs)
    """
    # Set model to evaluation mode
    # In eval mode, FP4QuantStubs will use their calibrated parameters to perform
    # quantization and dequantization on activations
    model_fp4_calibrated.eval()
    return model_fp4_calibrated

def prepare_model_for_qat(model_fp32: nn.Module, inplace: bool = False, learnable_qat_params: bool = False) -> nn.Module:
    """Prepares a float32 PyTorch model for Quantization-Aware Training (QAT) with IgniteFP4.
    
    This function is similar to `prepare_model_for_ptq()`, but configures the model for QAT:
    
    1. Replaces standard PyTorch layers with their FP4 counterparts (FP4Linear, FP4Conv2d, etc.)
    2. Inserts `FP4QuantStub` modules configured in QAT mode (`qat_mode=True`), meaning they will
       perform fake quantization (quantize then dequantize) during training.
    3. Optionally makes the quantization parameters (scale and zero-point) in these stubs learnable,
       allowing them to be optimized alongside model weights during training.
    
    The resulting model can be fine-tuned to adapt to the effects of FP4 quantization, potentially
    achieving better accuracy than post-training quantization alone.
    
    Args:
        model_fp32 (nn.Module): The original float32 PyTorch model.
        inplace (bool): If True, modify the model in-place. Otherwise, create a deep copy.
                       Default: False.
        learnable_qat_params (bool): If True, the scale and zero-point parameters in `FP4QuantStub` 
                                   modules become learnable parameters that are optimized during
                                   training. Default: False.
                                   
    Returns:
        nn.Module: The model prepared for FP4 QAT, with FP4 layers and QAT-configured `FP4QuantStub` modules.
        
    Example:
        >>> # Prepare model for QAT with learnable quantization parameters
        >>> qat_model = prepare_model_for_qat(float_model, learnable_qat_params=True)
        >>> # Move to device and set to training mode
        >>> qat_model.to('cuda')
        >>> qat_model.train()
        >>> # QAT training loop
        >>> optimizer = torch.optim.Adam(qat_model.parameters())
        >>> for epoch in range(num_epochs):
        >>>     for batch_idx, (inputs, targets) in enumerate(train_loader):
        >>>         outputs = qat_model(inputs)
        >>>         loss = criterion(outputs, targets)
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         optimizer.zero_grad()
    """
    if not inplace:
        model_fp4 = copy.deepcopy(model_fp32)
    else:
        # In-place modification warning - same as in prepare_model_for_ptq()
        print("Warning: `inplace=True` is not fully robust yet. Using deepcopy.")
        model_fp4 = copy.deepcopy(model_fp32)

    # Recursive preparation, setting qat_mode=True and passing the learnable_params flag
    _recursive_prepare(model_fp4, "", qat_mode=True, learnable_params=learnable_qat_params)
    return model_fp4

def convert_qat_model_to_eval(model_fp4_qat: nn.Module) -> nn.Module:
    """Converts a QAT-trained model to evaluation mode for FP4-simulated inference.
    
    After a model has been trained with Quantization-Aware Training (QAT), this function
    prepares it for simulated FP4 inference:
    
    1. Sets the model to evaluation mode (`.eval()`). This changes the behavior of modules
       during forward passes:
       - `FP4QuantStub` modules will perform actual quantization using their parameters
         (which may have been learned during QAT if `learnable_qat_params=True` was used).
       - Batch normalization layers will use their running statistics instead of batch statistics.
       - Dropout layers will not drop activations.
    
    The conversion is primarily behavioral; no structural changes are made to the model.
    
    Args:
        model_fp4_qat (nn.Module): A model that has been prepared for QAT and trained.
        
    Returns:
        nn.Module: The same model, now in evaluation mode ready for FP4-simulated inference.
        
    Example:
        >>> # After QAT training
        >>> eval_model = convert_qat_model_to_eval(qat_model)
        >>> # Now ready for FP4-simulated inference
        >>> with torch.no_grad():
        >>>     outputs = eval_model(test_inputs)
    """
    # Optional: Finalize any per-layer operations needed before converting to eval
    # For example, if needed, could re-compute quantization parameters for any non-learnable stubs
    # based on latest statistics.
    
    # Set model to evaluation mode
    # In eval mode:
    # - FP4QuantStubs will use their parameters (learned or fixed) to quantize/dequantize
    # - BatchNorm layers use running statistics
    # - Dropout layers don't drop activations
    model_fp4_qat.eval()
    return model_fp4_qat

# More functions to come: convert_to_eval_model etc. 