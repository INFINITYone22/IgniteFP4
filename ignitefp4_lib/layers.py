# ignitefp4_lib/layers.py
"""
Defines custom PyTorch layers that simulate FP4 quantization for weights and/or activations.

This module includes:
- `FP4Linear`: A linear layer with FP4-simulated weights.
- `FP4Conv2d`: A 2D convolutional layer with FP4-simulated weights.
- `FP4QuantStub`: A module to observe activation statistics and perform fake FP4 quantization
  for activations. Supports different modes for PTQ (calibration) and QAT (fixed or learnable parameters).
- `FP4DequantStub`: A placeholder module, currently an identity operation.
- `FP4BatchNorm2d`: A BatchNorm2d layer with FP4-simulated affine parameters.
- `FP4AvgPool2d`, `FP4MaxPool2d`: Wrappers for standard pooling layers.
- `FP4ConvBnReLU2d`: A fused layer combining FP4Conv2d, FP4BatchNorm2d, and ReLU.

These layers are designed to be integrated into standard PyTorch models to simulate
the effects of FP4 precision during training (QAT) or post-training (PTQ).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .numerics import (
    quantize_to_fp4_symmetric, dequantize_from_fp4_symmetric,
    calculate_asymmetric_scale_zeropoint,
    quantize_to_fp4_asymmetric, quantize_to_fp4_asymmetric_ste,
    dequantize_from_fp4_asymmetric,
    FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL,
    FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL
)

class FP4Linear(nn.Module):
    """A PyTorch `nn.Linear` layer that simulates 4-bit precision for its weights.

    The weights are quantized to a simulated 4-bit format using signed symmetric quantization.
    This means the real value 0.0 maps to quantized 0, and the scale is determined by the
    maximum absolute weight value. The bias term, if present, is kept in full floating-point precision.

    During the `forward` pass:
    1. The 4-bit quantized weights (`fp4_weight`) are dequantized back to floating-point
       using their associated `weight_scale`.
    2. The standard `torch.nn.functional.linear` operation is performed with these
       dequantized weights and the input tensor.

    This layer is primarily used for two purposes:
    - **Simulating FP4 Inference:** To understand the performance and accuracy impact of using
      FP4 weights in a model that was originally trained in float.
    - **Quantization-Aware Training (QAT):** To model the effect of FP4 weight quantization
      during the training process, allowing the model to adapt to the precision loss.

    The actual 4-bit weights are stored as `torch.int8` for convenience, as PyTorch does not
    have a native 4-bit integer type. The values within this `int8` tensor will be
    in the range [-8, 7], corresponding to the 16 levels of signed 4-bit representation.

    Args:
        in_features (int): Size of each input sample (number of input features).
        out_features (int): Size of each output sample (number of output features).
        bias (bool): If ``True``, adds a learnable bias parameter to the output. Default: ``True``.
        device (optional): The target device for the layer's parameters and buffers (e.g., 'cpu', 'cuda').
        dtype (optional): The desired floating-point type for parameters and buffers (e.g., `torch.float32`).

    Attributes:
        in_features (int): Stores `in_features`.
        out_features (int): Stores `out_features`.
        fp4_weight (torch.Tensor): A buffer storing the 4-bit quantized weights. 
                                   Shape: (`out_features`, `in_features`). Dtype: `torch.int8`.
                                   Values are in the range [-8, 7].
        weight_scale (torch.Tensor): A buffer storing the scalar, per-tensor scale factor used to quantize
                                     and dequantize `fp4_weight`. Shape: (1,). Dtype: `torch.float32` (or as specified).
        bias (torch.nn.Parameter, optional): The learnable bias parameter of shape (`out_features`).
                                            If `bias` is ``False`` during initialization, this is ``None``.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Register buffers for quantized weights and their scale.
        # These are not nn.Parameters because their values are typically derived from full-precision weights
        # or fixed after calibration, not directly learned via backpropagation in the same way as standard weights.
        # However, the scale *could* be made learnable in advanced QAT scenarios, but that would require
        # a different setup, perhaps involving nn.Parameter for scale and custom backward hooks.
        self.register_buffer('fp4_weight', torch.empty((out_features, in_features), dtype=torch.int8, device=device))
        # Scale is typically float32 for precision, even if layer dtype is float16/bfloat16.
        self.register_buffer('weight_scale', torch.empty(1, dtype=torch.float32, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            # Important to register as None if no bias, for PyTorch's state_dict and module saving/loading.
            self.register_parameter('bias', None)
        
        # Initialize bias (if any). Weights are not initialized here directly as they are loaded
        # from a float module and then quantized.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the bias parameter using a Kaiming uniform-like distribution.
        
        The weights (`fp4_weight` and `weight_scale`) are not initialized here as they are
        expected to be set by `load_from_float_linear`. This method only handles bias.
        If a custom initialization for quantized weights were needed, it would be more complex,
        potentially involving initializing float weights, quantizing them, and then storing.
        """
        if self.bias is not None:
            # Kaiming uniform initialization for bias, similar to nn.Linear.
            # The fan_in is based on the conceptual float weight's shape.
            # self.fp4_weight has the same shape (out_features, in_features) as the original float weight.
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fp4_weight) 
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_from_float_linear(self, float_linear_module: nn.Linear):
        """Loads parameters from a standard `torch.nn.Linear` module and quantizes its weights.
        
        The `weight` tensor from `float_linear_module` is quantized using signed symmetric FP4
        quantization. The resulting 4-bit integer weights (stored as int8) and their
        corresponding scale factor are stored in `self.fp4_weight` and `self.weight_scale`.
        The `bias` from `float_linear_module` (if present) is copied directly.

        This method is crucial for:
        - Post-Training Quantization (PTQ): Converting a pre-trained float model to an FP4-simulated model.
        - QAT Initialization: Initializing an `FP4Linear` layer with pre-trained float weights before QAT.

        Args:
            float_linear_module (nn.Linear): The source `nn.Linear` module whose parameters
                                             will be loaded and (weights) quantized.
        Returns:
            self: The current `FP4Linear` instance, allowing for chained calls.
        """
        if not isinstance(float_linear_module, nn.Linear):
            raise TypeError("Input module must be an instance of torch.nn.Linear to load parameters.")

        float_weight = float_linear_module.weight.data
        
        # Ensure the float weight is on the same device as this module's buffers before quantization.
        # This is important if `float_linear_module` is on a different device.
        current_device = self.fp4_weight.device # Device of this FP4Linear layer's buffers
        float_weight_on_device = float_weight.to(current_device)

        # Quantize the float weights to FP4 symmetric format.
        # `quantize_to_fp4_symmetric` returns the int8 quantized tensor and the float scale.
        quantized_w, scale_w = quantize_to_fp4_symmetric(float_weight_on_device)
        
        # Store the quantized weights and their scale in the layer's buffers.
        self.fp4_weight.data.copy_(quantized_w) # `quantized_w` is already on `current_device`
        # Ensure scale is also on the correct device and potentially dtype.
        self.weight_scale.data.copy_(scale_w.to(dtype=self.weight_scale.dtype, device=current_device)) 

        # Handle the bias term.
        if float_linear_module.bias is not None:
            if self.bias is not None:
                # Copy bias data, ensuring it's on the correct device.
                self.bias.data.copy_(float_linear_module.bias.data.to(current_device))
            else:
                # This FP4Linear layer was initialized with bias=False, but the source float layer has a bias.
                print(f"Warning: Source nn.Linear (in_features={float_linear_module.in_features}, "
                      f"out_features={float_linear_module.out_features}) has a bias, but this FP4Linear "
                      f"(in_features={self.in_features}, out_features={self.out_features}) "
                      f"was initialized with bias=False. Source bias will not be loaded.")
        elif self.bias is not None:
            # This FP4Linear layer expects a bias, but the source float layer does not have one.
            # The bias in FP4Linear will remain as initialized by reset_parameters().
            print(f"Warning: Source nn.Linear (in_features={float_linear_module.in_features}, "
                  f"out_features={float_linear_module.out_features}) does not have a bias, but this FP4Linear "
                  f"(in_features={self.in_features}, out_features={self.out_features}) expects one. "
                  f"FP4Linear's bias will retain its initial values.")
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass using FP4-simulated weights.
        
        The core steps are:
        1. Ensure the input tensor `x` is on the same device as the layer's parameters.
        2. Dequantize `self.fp4_weight` to a floating-point tensor using `self.weight_scale`.
           The dequantized weight will have the dtype of `self.weight_scale`.
        3. Perform the standard linear operation `F.linear(x, dequantized_weight, self.bias)`.

        Args:
            x (torch.Tensor): The input tensor. Expected shape is (N, ..., `in_features`), where N is
                              the batch size and ... represents any number of additional dimensions.
        Returns:
            torch.Tensor: The output tensor of shape (N, ..., `out_features`).
        """
        # Ensure input tensor `x` is on the same device as the layer's parameters/buffers.
        # self.weight_scale.device is a reliable way to get the layer's current device.
        current_device = self.weight_scale.device
        if x.device != current_device:
            x = x.to(current_device)

        # Dequantize the 4-bit weights back to floating point before the multiplication.
        # The dtype of dequantized_weight will match self.weight_scale.dtype (typically float32).
        dequantized_weight = dequantize_from_fp4_symmetric(self.fp4_weight, self.weight_scale)
        
        # Perform the linear operation using the dequantized weights.
        # The bias, if present, is already in float.
        output = F.linear(x, dequantized_weight, self.bias)
        return output

    def extra_repr(self) -> str:
        """Provides a string representation with key layer parameters for printing the model summary."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, weight_quantized=FP4_symmetric'

# Example Usage (for quick testing, to be formalized in test_layers.py)
if __name__ == '__main__':
    # Create a standard float linear layer
    float_lin = nn.Linear(20, 30)

    # Create our FP4Linear layer
    fp4_lin = FP4Linear(20, 30)
    
    # Load (and quantize) weights from the float layer
    fp4_lin.load_from_float_linear(float_lin)

    print("FP4Linear Layer:", fp4_lin)
    print("Original float weight sample:", float_lin.weight.data[0, :5])
    print("Stored FP4 weight sample (int8):", fp4_lin.fp4_weight[0, :5])
    print("Weight scale:", fp4_lin.weight_scale.item())

    # Test forward pass
    input_tensor = torch.randn(1, 20)
    output_float = float_lin(input_tensor)
    output_fp4_sim = fp4_lin(input_tensor)

    print("\nOutput from original float layer:", output_float)
    print("Output from FP4 simulated layer:", output_fp4_sim)
    print("Difference (max abs):", torch.abs(output_float - output_fp4_sim).max().item())
    
    # Test moving module to device
    if torch.cuda.is_available():
        print("\nTesting device transfer (CUDA)...")
        device = torch.device("cuda")
        
        float_lin_cuda = nn.Linear(5,2).to(device)
        # Create on CPU then move, or create directly on device
        fp4_lin_cuda = FP4Linear(5,2, device=device)
        fp4_lin_cuda.load_from_float_linear(float_lin_cuda)
        
        assert fp4_lin_cuda.fp4_weight.device.type == "cuda", "FP4 weight not on CUDA"
        assert fp4_lin_cuda.weight_scale.device.type == "cuda", "Weight scale not on CUDA"
        if fp4_lin_cuda.bias is not None:
            assert fp4_lin_cuda.bias.device.type == "cuda", "Bias not on CUDA"

        input_cuda = torch.randn(3,5).to(device)
        output_cuda_fp4 = fp4_lin_cuda(input_cuda)
        print("CUDA FP4 output device:", output_cuda_fp4.device)
        assert output_cuda_fp4.device.type == "cuda", "Output not on CUDA"
        print("CUDA tests passed (basic device check and forward pass).")
    else:
        print("\nCUDA not available, skipping device transfer test.")

class FP4Conv2d(nn.Module):
    """A PyTorch `nn.Conv2d` layer that simulates 4-bit precision for its weights.

    Similar to `FP4Linear`, the weights of this 2D convolutional layer are quantized to
    a simulated 4-bit format using signed symmetric quantization (per-tensor scale).
    The bias term, if present, remains in full floating-point precision.

    During the `forward` pass:
    1. The 4-bit quantized weights (`fp4_weight`) are dequantized back to floating-point
       using their associated `weight_scale`.
    2. The standard `torch.nn.functional.conv2d` operation is performed with these
       dequantized weights and the input tensor, along with other convolution parameters
       (stride, padding, etc.).

    This layer is intended for:
    - Simulating FP4 inference for convolutional neural networks (CNNs).
    - Quantization-Aware Training (QAT) for CNNs, modeling FP4 weight effects.

    The 4-bit weights are stored as `torch.int8` (values in [-8, 7]).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel (e.g., 3 or (3, 3)).
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
        padding_mode (str, optional): Type of padding: ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
        device (optional): The target device for the layer's parameters and buffers.
        dtype (optional): The desired floating-point type for parameters and buffers.

    Attributes:
        in_channels (int): Stores `in_channels`.
        out_channels (int): Stores `out_channels`.
        kernel_size (tuple): Stores `kernel_size` (always stored as a tuple).
        stride (tuple): Stores `stride`.
        padding (tuple or str): Stores `padding`.
        dilation (tuple): Stores `dilation`.
        groups (int): Stores `groups`.
        padding_mode (str): Stores `padding_mode`.
        fp4_weight (torch.Tensor): Buffer for 4-bit quantized weights (as `torch.int8`).
                                   Shape: (`out_channels`, `in_channels` // `groups`, `kernel_size[0]`, `kernel_size[1]`).
        weight_scale (torch.Tensor): Buffer for the per-tensor scale factor of `fp4_weight`.
                                     Shape: (1,). Dtype: `torch.float32` (or as specified).
        bias (torch.nn.Parameter, optional): The learnable bias. Shape: (`out_channels`). ``None`` if `bias=False`.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size, # Can be int or tuple
                 stride = 1,  # Can be int or tuple
                 padding = 0, # Can be int, tuple or str
                 dilation = 1,# Can be int or tuple
                 groups: int = 1, 
                 bias: bool = True,
                 padding_mode: str = 'zeros', # 'zeros', 'reflect', 'replicate', 'circular'
                 device=None, 
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # Store configuration, converting single int args to tuples for consistency (like nn.Conv2d does)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = padding # Can be string like 'same' or int/tuple
        self.dilation = nn.modules.utils._pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Determine the shape of the weight tensor for nn.Conv2d
        # (out_channels, in_channels // groups, kernel_height, kernel_width)
        weight_shape = (out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        
        self.register_buffer('fp4_weight', torch.empty(weight_shape, dtype=torch.int8, device=device))
        self.register_buffer('weight_scale', torch.empty(1, dtype=torch.float32, device=device)) # Per-tensor scale

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the bias parameter using Kaiming uniform initialization (if bias is enabled).
        Weights (`fp4_weight`, `weight_scale`) are intended to be loaded via `load_from_float_conv2d`.
        """
        if self.bias is not None:
            # Bias initialization similar to nn.Conv2d
            # fan_in is calculated based on the conceptual float weight shape
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fp4_weight) 
            if fan_in > 0:
                bound = 1 / (fan_in**0.5)
                nn.init.uniform_(self.bias, -bound, bound)
            else: # Handle cases like fan_in = 0, e.g. kernel_size is 0 or groups mismatch somehow
                nn.init.zeros_(self.bias)

    def load_from_float_conv2d(self, float_conv_module: nn.Conv2d):
        """Loads parameters from a `torch.nn.Conv2d` module and quantizes its weights.

        The `weight` from `float_conv_module` is quantized to FP4 symmetric format (per-tensor scale).
        The `bias` (if present) is copied directly.

        Args:
            float_conv_module (nn.Conv2d): The source `nn.Conv2d` module.
        Returns:
            self: The current `FP4Conv2d` instance.
        """
        if not isinstance(float_conv_module, nn.Conv2d):
            raise TypeError("Input module must be an instance of torch.nn.Conv2d to load parameters.")

        float_weight = float_conv_module.weight.data
        current_device = self.fp4_weight.device # Device of this FP4Conv2d layer's buffers
        float_weight_on_device = float_weight.to(current_device)

        quantized_w, scale_w = quantize_to_fp4_symmetric(float_weight_on_device)
        
        self.fp4_weight.data.copy_(quantized_w)
        self.weight_scale.data.copy_(scale_w.to(dtype=self.weight_scale.dtype, device=current_device))

        if float_conv_module.bias is not None:
            if self.bias is not None:
                self.bias.data.copy_(float_conv_module.bias.data.to(current_device))
            else:
                print(f"Warning: Source nn.Conv2d (in_C={float_conv_module.in_channels}, "
                      f"out_C={float_conv_module.out_channels}, K={float_conv_module.kernel_size}) has a bias, "
                      f"but this FP4Conv2d (in_C={self.in_channels}, out_C={self.out_channels}, K={self.kernel_size}) "
                      f"was initialized with bias=False. Source bias will not be loaded.")
        elif self.bias is not None:
            print(f"Warning: Source nn.Conv2d (in_C={float_conv_module.in_channels}, "
                  f"out_C={float_conv_module.out_channels}, K={float_conv_module.kernel_size}) does not have a bias, "
                  f"but this FP4Conv2d (in_C={self.in_channels}, out_C={self.out_channels}, K={self.kernel_size}) expects one. "
                  f"FP4Conv2d's bias will retain its initial values.")
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass using FP4-simulated weights.

        Steps:
        1. Ensure input `x` is on the correct device.
        2. Dequantize `self.fp4_weight` to float using `self.weight_scale`.
        3. Perform `F.conv2d` with dequantized weights and other conv parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (N, `in_channels`, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (N, `out_channels`, H_out, W_out).
        """
        current_device = self.weight_scale.device
        if x.device != current_device:
            x = x.to(current_device)

        dequantized_weight = dequantize_from_fp4_symmetric(self.fp4_weight, self.weight_scale)
        
        # Handle string padding case for F.conv2d by calculating it if necessary
        # Note: This is a simplified handling. For production, use PyTorch's internal _padding_repeated_twice if available
        # or ensure padding is pre-calculated if it's a string like 'same' or 'valid'.
        # For now, assuming integer/tuple padding is mostly used or 'same' is handled by some higher utility if needed.
        # F.conv2d itself can handle string padding if PyTorch version is recent enough and mode allows.
        current_padding = self.padding
        # if isinstance(self.padding, str):
        #     # Basic handling for 'same'. More robust calculation might be needed for various strides/dilations.
        #     # This is a complex topic. For now, we rely on F.conv2d to handle string padding if supported.
        #     pass 

        output = F.conv2d(x, dequantized_weight, self.bias, 
                          self.stride, current_padding, self.dilation, self.groups)
        return output

    def extra_repr(self) -> str:
        """Provides a string representation with key layer parameters."""
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        # Padding can be int, tuple, or string. nn.Conv2d stores it as a tuple after init usually.
        # Only show if not default (0 for int/tuple, or if string)
        if isinstance(self.padding, str) or (not isinstance(self.padding, str) and self.padding != (0,0) and self.padding != 0):
             s += ', padding={padding}'
        if self.dilation != (1, 1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        # padding_mode is only relevant if padding is non-zero
        if self.padding_mode != 'zeros' and self.padding != 0 and self.padding != (0,0):
            s += ', padding_mode={padding_mode}'
        s += ', weight_quantized=FP4_symmetric'
        # Use self.__dict__ might be too verbose, let's try specific attributes
        # For simplicity, we can rely on the parameters passed to init being stored
        # or construct a dict for formatting.
        format_dict = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
            'padding_mode': self.padding_mode
        }
        return s.format(**format_dict) 

class FP4QuantStub(nn.Module):
    """A module that simulates 4-bit precision for activations (tensor data).

    This stub serves three main purposes depending on the mode it's in:
    1. **Calibration Mode** (PTQ, Post-Training Quantization): Observes activation statistics
       across batches to determine appropriate scale/zero-point for quantization.
    2. **Eval Mode with Fixed Parameters**: Performs quantization and dequantization using 
       the parameters determined during calibration (for simulating FP4 inference).
    3. **Quantization-Aware Training (QAT) Mode**: Performs "fake quantization" (quantize then
       dequantize) during training to let the model adapt to quantization effects. The quantization
       parameters can be either fixed or learnable.

    The activation quantization is asymmetric (unsigned), meaning it maps the full observed range
    [min, max] to the 4-bit range [0, 15] using both a scale and a zero-point. This is well-suited
    for activations that often have non-zero minimum values (especially after ReLU).

    Args:
        num_bits (int): The number of bits for quantization. Default: 4.
                        Currently, only 4-bit is actively supported.
        observer_type (str): Type of observer for determining quantization parameters.
                            Currently only 'minmax' is implemented, which tracks min and max values.
        qat_mode (bool): If True, enables Quantization-Aware Training mode (fake quantization).
                         If False, the module is in calibration/PTQ mode. Default: False.
        learnable_params (bool): If True and in QAT mode, the scale and zero-point parameters
                                become learnable with gradients. This allows the model to optimize
                                quantization parameters during training. Default: False.
        device (optional): The target device (e.g., 'cpu', 'cuda').
        dtype (optional): The desired floating-point type (e.g., `torch.float32`).

    Attributes:
        num_bits (int): Number of bits for quantization (e.g., 4 for FP4).
        qat_mode (bool): Whether the module is in QAT mode (True) or PTQ/calibration mode (False).
        learnable_params (bool): Whether quantization parameters can be learned during QAT.
        observer_type (str): The type of observer being used (e.g., 'minmax').
        activation_scale (torch.Tensor or nn.Parameter): Scale factor for activation quantization.
                                                       Shape: (1,). A Parameter if learnable.
        activation_zp (torch.Tensor or nn.Parameter): Zero-point for activation quantization.
                                                     Shape: (1,). A Parameter if learnable.
        min_val (torch.Tensor): Tracks minimum observed activation value during calibration.
                               Shape: (1,).
        max_val (torch.Tensor): Tracks maximum observed activation value during calibration.
                               Shape: (1,).
        num_batches_tracked (torch.Tensor): Counts batches seen during calibration.
                                           Shape: (1,). Dtype: `torch.int32`.
    """
    def __init__(self, num_bits: int = 4, observer_type: str = 'minmax', 
                 qat_mode: bool = False, learnable_params: bool = False, 
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Store configuration
        self.num_bits = num_bits
        if num_bits != 4:
            raise NotImplementedError(f"Currently only 4-bit is supported, got {num_bits}")
        
        self.qat_mode = qat_mode
        self.learnable_params = learnable_params
        self.observer_type = observer_type
        
        if observer_type != 'minmax':
            raise NotImplementedError(f"Currently only 'minmax' observer is supported, got {observer_type}")
        
        # For observation/calibration (tracking range statistics)
        self.register_buffer('min_val', torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val', torch.tensor(float('-inf'), **factory_kwargs))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.int32, device=device))
        
        # For quantization (either fixed after calibration or learnable during QAT)
        if learnable_params and qat_mode:
            # For QAT with learnable parameters, use nn.Parameter so these get optimized
            # Scale is initialized to 1.0 (will be updated once we see data)
            self.activation_scale = nn.Parameter(torch.tensor(1.0, **factory_kwargs))
            # Zero-point is initialized to 0 (will be updated once we see data)
            self.activation_zp = nn.Parameter(torch.tensor(0, dtype=torch.float32, **factory_kwargs))
        else:
            # For PTQ or QAT with fixed parameters, use buffers
            self.register_buffer('activation_scale', torch.tensor(1.0, **factory_kwargs))
            self.register_buffer('activation_zp', 
                                torch.tensor(0, dtype=torch.int32 if not qat_mode else torch.float32, 
                                            device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that performs statistics gathering and/or quantization based on mode.

        Behavior depends on training mode and configuration:
        1. **Training Mode, QAT Disabled**: Only collects statistics for calibration.
        2. **Training Mode, QAT Enabled**: Performs fake quantization (quantize+dequantize)
           using current parameters (fixed or being learned).
        3. **Eval Mode**: Always quantizes and dequantizes, simulating the quantization
           impact during inference.

        Args:
            x (torch.Tensor): Input tensor (activations) to be observed or quantized.
        
        Returns:
            torch.Tensor: If in training mode without QAT, returns the input unchanged.
                         If QAT or eval mode, returns the dequantized tensor after quantization.
        """
        # Move input to the same device as the module's parameters
        current_device = self.min_val.device
        if x.device != current_device:
            x = x.to(current_device)
            
        # In training mode, we either gather statistics or do QAT fake quantization
        if self.training:
            # QAT mode: quantize then dequantize to simulate quantization impact
            if self.qat_mode:
                # If we have learnable parameters, ensure they're in the proper format
                # for the STE quantization function (which expects float)
                if self.learnable_params:
                    # Tensors are already nn.Parameters and thus already float
                    scale_for_quant = self.activation_scale
                    zp_for_quant = self.activation_zp
                else:
                    # Use existing buffers, ensuring zero-point is float for QAT
                    scale_for_quant = self.activation_scale
                    zp_for_quant = self.activation_zp  # Should already be float for QAT
                
                # In QAT, we use the STE version for gradients to flow through rounding
                quantized = quantize_to_fp4_asymmetric_ste(x, scale_for_quant, zp_for_quant)
                result = dequantize_from_fp4_asymmetric(quantized, scale_for_quant, zp_for_quant)
                return result
            
            # Not QAT but training: record min/max for calibration
            # (this only updates statistics, doesn't modify the tensor)
            with torch.no_grad():  # No need for gradients when just collecting stats
                # Update observed min/max
                self.min_val = torch.min(torch.min(x), self.min_val)
                self.max_val = torch.max(torch.max(x), self.max_val)
                # Track batch count for averaging if needed
                self.num_batches_tracked += 1
                
            # During normal training without QAT, return x unchanged
            return x
            
        # In eval mode, always simulate quantization (for both PTQ and QAT)
        else:
            # Regular quantization (not via STE since we're in eval mode)
            # Zero-point should be int32 for actual quantization in eval
            if not isinstance(self.activation_zp, torch.Tensor) or self.activation_zp.dtype != torch.int32:
                # This can happen if coming from QAT mode where zp is float
                # We need to convert it to int32 for actual quantization
                zp_int = torch.round(self.activation_zp).to(torch.int32)
            else:
                zp_int = self.activation_zp
                
            quantized = quantize_to_fp4_asymmetric(x, self.activation_scale, zp_int)
            return dequantize_from_fp4_asymmetric(quantized, self.activation_scale, zp_int)

    def compute_quant_params(self):
        """Calculates the scale and zero-point parameters based on observed statistics.
        
        This is typically called:
        1. After PTQ calibration to finalize parameters before conversion to inference.
        2. At the beginning of QAT to initialize the quantization parameters.
        3. During QAT (if `learnable_params=False`) to update parameters with recent statistics.

        The scale and zero-point are computed from the min and max values observed during
        calibration using the asymmetric quantization equations.

        Returns:
            tuple: A tuple containing (scale, zero_point).
        """
        if self.min_val.item() == float('inf') or self.max_val.item() == float('-inf'):
            # No data has been observed yet
            print(f"Warning: No data has been observed by FP4QuantStub. "
                  f"min_val={self.min_val.item()}, max_val={self.max_val.item()}")
            # Return default values
            return self.activation_scale, self.activation_zp

        with torch.no_grad():
            # Build a representative tensor containing min and max values for calculation
            # This is a simple approach - for more sophisticated statistics, we might
            # use a running average or histogram-based approach
            repr_tensor = torch.tensor([self.min_val.item(), self.max_val.item()], 
                                       device=self.min_val.device,
                                       dtype=self.min_val.dtype)
            
            # Calculate scale and zero-point using the asymmetric quantization formula
            scale, zp = calculate_asymmetric_scale_zeropoint(repr_tensor, bits=self.num_bits)
            
            # Update the module's parameters
            if isinstance(self.activation_scale, nn.Parameter):
                # For QAT with learnable parameters, retain gradients (don't use .data.copy_)
                # No need to call .detach() since scale from calculate_asymmetric_scale_zeropoint
                # doesn't have gradients attached
                self.activation_scale.data.copy_(scale)
                # Zero-point is kept as float during QAT training, even if learnable
                self.activation_zp.data.copy_(zp.to(dtype=torch.float32)) 
            else:
                # For PTQ or QAT with fixed parameters, update buffers
                self.activation_scale.copy_(scale)
                
                # If in QAT mode with fixed parameters, store zero-point as float for STE
                if self.qat_mode:
                    self.activation_zp.copy_(zp.to(dtype=torch.float32))
                else:
                    # For PTQ, store as int32 (what will be used at evaluation time)
                    self.activation_zp.copy_(zp)
                
            return scale, zp

    def extra_repr(self) -> str:
        """Provides detailed string representation with key module parameters."""
        return f'num_bits={self.num_bits}, observer={self.observer_type}, qat_mode={self.qat_mode}, learnable_params={self.learnable_params}'

class FP4DequantStub(nn.Module):
    """A placeholder module for dequantization step in a quantized model.
    
    In the current implementation, this is an identity operation since the actual
    dequantization happens within the `FP4QuantStub.forward` method. This stub is
    included for symmetry with the `FP4QuantStub` and to maintain the structure
    expected by the quantization workflow.

    In a real hardware implementation, this would be where the 4-bit quantized values
    are converted back to floating-point for operations that can't be performed in
    the low-precision domain.
    
    No parameters or special configurations required for this stub.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - currently an identity operation.
        
        Args:
            x (torch.Tensor): Input tensor (already dequantized by `FP4QuantStub`)
            
        Returns:
            torch.Tensor: Unmodified input tensor
        """
        return x
    
    def extra_repr(self) -> str:
        """String representation of this module - no parameters to report."""
        return 'Identity dequantization stub'

class FP4BatchNorm2d(nn.Module):
    """A PyTorch `nn.BatchNorm2d` layer that simulates 4-bit precision for its affine parameters.

    Like standard BatchNorm, this normalizes inputs across the batch dimension and
    optionally applies affine transformation (scale and shift). Unlike standard BatchNorm,
    the affine parameters (weight and bias) are quantized to simulated 4-bit precision
    if `affine=True`.

    Weights (gamma) are quantized using signed symmetric FP4 quantization.
    Bias (beta) terms remain in full floating-point.
    
    Running mean and variance are maintained in full floating-point precision.

    This layer enables simulation of FP4-precision batch normalization, which is
    particularly important in models where batch norm parameters might be fused
    with convolution weights.

    Args:
        num_features (int): Number of features/channels to normalize over.
        eps (float): Small constant for numerical stability in normalization.
                    Default: 1e-5.
        momentum (float): Value for running_mean and running_var calculation.
                         Default: 0.1.
        affine (bool): If True, this module has learnable affine parameters (scale and shift).
                      Default: True.
        track_running_stats (bool): If True, this module tracks running mean and variance.
                                   Default: True.
        device (optional): The target device.
        dtype (optional): The desired floating-point type.

    Attributes:
        num_features (int): Number of features/channels.
        eps (float): Small constant for numerical stability.
        momentum (float): Value for running statistics calculation.
        affine (bool): Whether affine transformation is applied.
        track_running_stats (bool): Whether running statistics are tracked.
        fp4_weight (torch.Tensor, optional): Buffer for 4-bit quantized scale/gamma parameter.
                                           Shape: (`num_features`). Only present if `affine=True`.
        weight_scale (torch.Tensor, optional): Buffer for the per-tensor scale factor of `fp4_weight`.
                                             Shape: (1,). Only present if `affine=True`.
        bias (nn.Parameter, optional): The learnable shift/beta parameter. 
                                      Shape: (`num_features`). Only present if `affine=True`.
        running_mean (torch.Tensor, optional): Buffer tracking the running mean during training.
                                             Shape: (`num_features`). Only present if `track_running_stats=True`.
        running_var (torch.Tensor, optional): Buffer tracking the running variance during training.
                                            Shape: (`num_features`). Only present if `track_running_stats=True`.
        num_batches_tracked (torch.Tensor, optional): Counts the number of batches processed.
                                                    Shape: (1,). Only present if `track_running_stats=True`.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # For a BatchNorm layer, we quantize the weight (scale/gamma) parameter if affine is enabled
        if self.affine:
            # Register buffers for the quantized weight and its scale
            self.register_buffer('fp4_weight', torch.empty(num_features, dtype=torch.int8, device=device))
            self.register_buffer('weight_scale', torch.empty(1, dtype=torch.float32, device=device))
            # Bias is kept in full float precision
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # For tracking running statistics, similar to regular BatchNorm
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=device))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        """Resets the running statistics to their initial values.
        
        Only relevant if `track_running_stats=True`.
        """
        if self.track_running_stats:
            # Initialize/reset running statistics
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        """Resets all parameters and running statistics.
        
        This is called during initialization and can be called manually.
        Like nn.BatchNorm2d, weights are not initialized here but loaded from float_bn_module
        via load_from_float_batchnorm2d method.
        """
        self.reset_running_stats()
        if self.affine:
            # Initialize bias to zeros as in regular BatchNorm2d
            # Note: Weight (fp4_weight, weight_scale) will be loaded from a float module
            nn.init.zeros_(self.bias)

    def load_from_float_batchnorm2d(self, float_bn_module: nn.BatchNorm2d):
        """Loads parameters from a standard `torch.nn.BatchNorm2d` module and quantizes the weight.
        
        The weight/gamma parameter is quantized to FP4 symmetric format, while the bias/beta
        and running statistics are copied directly.

        Args:
            float_bn_module (nn.BatchNorm2d): Source BatchNorm2d module to load parameters from.
        
        Returns:
            self: The current `FP4BatchNorm2d` instance.
        """
        if not isinstance(float_bn_module, nn.BatchNorm2d):
            raise TypeError("Input module must be an instance of torch.nn.BatchNorm2d")
            
        # Check and copy running statistics if both modules track them
        if self.track_running_stats and float_bn_module.track_running_stats:
            current_device = self.running_mean.device
            self.running_mean.data.copy_(float_bn_module.running_mean.data.to(current_device))
            self.running_var.data.copy_(float_bn_module.running_var.data.to(current_device))
            self.num_batches_tracked.data.copy_(float_bn_module.num_batches_tracked.data.to(current_device))
        elif self.track_running_stats:
            print("Warning: Source BatchNorm2d does not track running stats, but this FP4BatchNorm2d does. "
                  "Running stats will retain their initial values.")
        elif float_bn_module.track_running_stats:
            print("Warning: Source BatchNorm2d tracks running stats, but this FP4BatchNorm2d does not. "
                  "Running stats will not be loaded.")
            
        # Check and copy/quantize affine parameters if both modules have them
        if self.affine and float_bn_module.affine:
            current_device = self.fp4_weight.device
            
            # Quantize the weight (gamma) parameter to FP4 symmetric
            float_weight = float_bn_module.weight.data.to(current_device)
            quantized_w, scale_w = quantize_to_fp4_symmetric(float_weight)
            
            self.fp4_weight.data.copy_(quantized_w)
            self.weight_scale.data.copy_(scale_w.to(dtype=self.weight_scale.dtype, device=current_device))
            
            # Copy bias (beta) parameter directly as we keep it in full float precision
            self.bias.data.copy_(float_bn_module.bias.data.to(current_device))
        elif self.affine:
            print("Warning: Source BatchNorm2d does not have affine parameters, but this FP4BatchNorm2d does. "
                  "Affine parameters will retain their initial values.")
        elif float_bn_module.affine:
            print("Warning: Source BatchNorm2d has affine parameters, but this FP4BatchNorm2d does not. "
                  "Affine parameters will not be loaded.")
            
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass using FP4-simulated weight (gamma) parameter.
        
        The core batch normalization calculation is the same as in `nn.BatchNorm2d`:
        1. In training mode, calculate batch statistics (mean and variance).
        2. Update running statistics if tracking them.
        3. Normalize the input using appropriate statistics.
        4. Apply the affine transformation if enabled, but using the dequantized weight.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where C = `num_features`.
            
        Returns:
            torch.Tensor: The normalized and transformed output of the same shape as the input.
        """
        # Ensure input is on the same device
        current_device = x.device
        if self.affine and self.fp4_weight.device != current_device:
            # Move module to the input's device if needed
            self.to(current_device)
        
        # If affine is enabled, dequantize the FP4 weight parameter
        if self.affine:
            dequantized_weight = dequantize_from_fp4_symmetric(self.fp4_weight, self.weight_scale)
        else:
            dequantized_weight = None
        
        # Use PyTorch's functional batch norm implementation
        # We pass our dequantized weight and regular bias for the affine transformation
        return F.batch_norm(
            x,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            dequantized_weight,  # Our dequantized FP4 weight (gamma)
            self.bias,           # Regular float bias (beta)
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps
        )

    def extra_repr(self) -> str:
        """Provides string representation with key module parameters."""
        return (f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}, '
                f'weight_quantized={self.affine and "FP4_symmetric" or "N/A"}')

class FP4AvgPool2d(nn.AvgPool2d):
    """A wrapper around nn.AvgPool2d to keep consistent naming in FP4-simulated models.
    
    This class directly inherits from `torch.nn.AvgPool2d` without changes to behavior.
    Average pooling is a mathematical operation that doesn't involve learned parameters,
    so there's nothing to quantize. It's included in the FP4 module family for API consistency
    when constructing fully FP4-simulated models.
    
    All parameters and behaviors are identical to `torch.nn.AvgPool2d`.
    """
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, 
                 count_include_pad=True, divisor_override=None):
        """Identical to nn.AvgPool2d initialization."""
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    
    def extra_repr(self) -> str:
        """Adds a 'non_quantized' marker to the string representation."""
        return super().extra_repr() + ', non_quantized=True (unchanged from nn.AvgPool2d)'

class FP4MaxPool2d(nn.MaxPool2d):
    """A wrapper around nn.MaxPool2d to keep consistent naming in FP4-simulated models.
    
    Like `FP4AvgPool2d`, this class directly inherits from `torch.nn.MaxPool2d` without
    changes to behavior. Max pooling is a selection operation without learned parameters,
    so there's nothing to quantize. It's included for API consistency.
    
    All parameters and behaviors are identical to `torch.nn.MaxPool2d`.
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        """Identical to nn.MaxPool2d initialization."""
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def extra_repr(self) -> str:
        """Adds a 'non_quantized' marker to the string representation."""
        return super().extra_repr() + ', non_quantized=True (unchanged from nn.MaxPool2d)'

class FP4ConvBnReLU2d(nn.Module):
    """A fused module combining FP4-simulated Conv2d, BatchNorm2d, and ReLU.
    
    This module represents a common pattern in CNNs: convolution, followed by batch normalization, 
    followed by ReLU activation. In this FP4-simulated version, the weights of both the convolution
    and batch normalization are quantized to simulated 4-bit precision.
    
    The forward pass:
    1. Performs a convolution with FP4-simulated weights
    2. Applies batch normalization with FP4-simulated affine parameters
    3. Applies a ReLU activation
    
    This fused module is useful for understanding the combined impact of FP4 precision
    on this common layer sequence. In hardware/compiler implementations, these operations
    are often fused for efficiency.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding for the convolution. Default: 0.
        dilation (int or tuple, optional): Dilation factor for convolution. Default: 1.
        groups (int, optional): Number of blocked connections from input to output channels. Default: 1.
        conv_bias (bool, optional): If True, adds a learnable bias to the convolution output.
                                  Typically False when followed by BatchNorm. Default: False.
        padding_mode (str, optional): Padding mode for convolution. Default: 'zeros'.
        eps (float, optional): Small constant for BatchNorm numerical stability. Default: 1e-5.
        momentum (float, optional): BatchNorm momentum for running statistics. Default: 0.1.
        relu_inplace (bool, optional): Whether to perform ReLU operation in-place. Default: False.
        device (optional): The target device for parameters.
        dtype (optional): The desired floating-point type for parameters.
    
    Attributes:
        conv (FP4Conv2d): The FP4-simulated convolutional layer.
        bn (FP4BatchNorm2d): The FP4-simulated batch normalization layer.
        relu (nn.ReLU): The ReLU activation function.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1,
                 groups: int = 1, 
                 conv_bias: bool = False, 
                 padding_mode: str = 'zeros',
                 eps: float = 1e-5, 
                 momentum: float = 0.1,
                 relu_inplace: bool = False,
                 device=None, 
                 dtype=None):
        super().__init__()
        
        # Sequential composition of FP4Conv2d, FP4BatchNorm2d, and ReLU
        # Note: It's typical to use bias=False for the conv layer when followed by batch norm
        self.conv = FP4Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                             dilation, groups, conv_bias, padding_mode, device, dtype)
        
        self.bn = FP4BatchNorm2d(out_channels, eps, momentum, True, True, device, dtype)
        
        self.relu = nn.ReLU(inplace=relu_inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the fused forward pass: conv -> bn -> relu.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H_in, W_in)
            
        Returns:
            torch.Tensor: Processed tensor of shape (N, out_channels, H_out, W_out)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def load_from_float_modules(self, float_conv_module: nn.Conv2d, float_bn_module: nn.BatchNorm2d):
        """Loads parameters from standard float `nn.Conv2d` and `nn.BatchNorm2d` modules.
        
        This method quantizes the weights of both the convolution and batch normalization
        components using their respective loading methods.
        
        Args:
            float_conv_module (nn.Conv2d): The source float convolution module.
            float_bn_module (nn.BatchNorm2d): The source float batch normalization module.
            
        Returns:
            self: The current `FP4ConvBnReLU2d` instance.
        """
        # Check that the modules are compatible
        if not isinstance(float_conv_module, nn.Conv2d) or not isinstance(float_bn_module, nn.BatchNorm2d):
            raise TypeError("Input modules must be instances of torch.nn.Conv2d and torch.nn.BatchNorm2d")
        
        # Ensure the dimensions match between the conv output and bn input
        if float_conv_module.out_channels != float_bn_module.num_features:
            raise ValueError(f"Conv out_channels ({float_conv_module.out_channels}) must match BN num_features ({float_bn_module.num_features})")
        
        # Load parameters from the float modules
        self.conv.load_from_float_conv2d(float_conv_module)
        self.bn.load_from_float_batchnorm2d(float_bn_module)
        
        return self
    
    def extra_repr(self) -> str:
        """Provides a concise string representation with key module parameters."""
        s = (f'in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, '
             f'kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, '
             f'relu_inplace={self.relu.inplace}')
        return s 