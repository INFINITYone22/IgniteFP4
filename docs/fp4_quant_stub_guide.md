# FP4QuantStub: A Guide to Activation Quantization

The `FP4QuantStub` module in `ignitefp4_lib.layers` is a cornerstone for simulating FP4 quantization of activations. It's a versatile `nn.Module` that behaves differently based on its configuration and the model's training state (`model.train()` or `model.eval()`). This guide details its modes of operation.

## Core Purpose

`FP4QuantStub` is inserted into a model (typically before FP4-simulated layers like `FP4Linear` or `FP4Conv2d`) to handle the quantization of input activations. It supports workflows for both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

Key attributes that control its behavior:
*   `qat_mode` (bool): Determines if the stub operates in QAT mode (True) or PTQ calibration/pass-through mode (False) during `model.train()`.
*   `learnable_params` (bool): If True (and `qat_mode` is True), the `scale` and `zero_point` for quantization become `nn.Parameter`s, learnable during QAT.
*   `training` (bool, inherited from `nn.Module`): The state of the parent model (`model.train()` or `model.eval()`).
*   `is_calibrated` (bool): A flag set to True after `compute_quant_params()` successfully calculates or initializes the `scale` and `zero_point`.
*   `enabled` (bool): If False, the stub acts as an identity function.

## Modes of Operation

Here's a breakdown of how `FP4QuantStub` functions in different scenarios:

### 1. PTQ Calibration Mode

*   **Configuration**:
    *   `model.train()` is active.
    *   `qat_mode = False` (default when `prepare_model_for_ptq` is used).
    *   `learnable_params = False` (default).
*   **Behavior**:
    *   **Observe Statistics**: The stub monitors the minimum and maximum values of the input tensor `x` across all batches of calibration data passed through it.
    *   **Pass-Through**: The input tensor `x` is passed through *unmodified*. No quantization is applied at this stage.
    *   **`compute_quant_params()`**: After the calibration data has been processed, this method is called (e.g., by `calibrate_model`). It uses the globally observed `min_val` and `max_val` to calculate the FP4 `scale` (float) and `zero_point` (int32) for asymmetric quantization. These are stored as buffers in the stub, and `is_calibrated` is set to `True`.

### 2. QAT Mode - Fixed/Batch Statistics (Non-Learnable Parameters)

*   **Configuration**:
    *   `model.train()` is active.
    *   `qat_mode = True` (set by `prepare_model_for_qat`).
    *   `learnable_params = False`.
*   **Behavior**:
    *   **Observe Statistics**: Continues to observe and update `min_val` and `max_val` from input batches.
    *   **Fake Quantization**: Performs fake quantization (quantize to simulated FP4, then dequantize back to float).
        *   If `is_calibrated = True` (meaning `compute_quant_params()` was called, perhaps after an initial calibration epoch or based on pre-set values): It uses its stored `scale` and `zero_point` buffers for the fake quantization.
        *   If `is_calibrated = False`: It can calculate `scale` and `zero_point` on-the-fly using its running `min_val` and `max_val` (or current batch stats if `min_val`/`max_val` are still at defaults). This is less stable for QAT from scratch but provides a dynamic approach.
    *   **Rounding**: Uses standard `torch.round()` for the quantization step.
    *   **`compute_quant_params()`**: Can be called (typically at the end of QAT, or periodically) to update the `scale` and `zero_point` buffers using the latest observed `min_val` and `max_val` from the training data. This effectively makes the quantization parameters adapt to the data distribution seen during QAT.

### 3. QAT Mode - Learnable Parameters

*   **Configuration**:
    *   `model.train()` is active.
    *   `qat_mode = True`.
    *   `learnable_params = True` (set if `learnable_qat_params=True` in `prepare_model_for_qat`).
*   **Behavior**:
    *   **Observe Statistics**: Still observes `min_val` and `max_val`, primarily for monitoring or potential re-initialization via `compute_quant_params()`.
    *   **Learnable `scale` and `zero_point`**: The `scale` and `zero_point` are `nn.Parameter`s. Their initial values can be set by calling `compute_quant_params()` once after observing some initial data, or they can start with default values.
    *   **Fake Quantization with STE**: Performs fake quantization using its learnable `scale` and `zero_point`.
    *   **Rounding**: Uses `_ste_round()` (Straight-Through Estimator for rounding). This is critical because standard rounding has zero gradients almost everywhere, which would prevent `scale` and `zero_point` from being updated during backpropagation. STE allows gradients to flow through the rounding operation as if it were an identity function in the backward pass.
    *   **`compute_quant_params()`**: If called, this method will *re-initialize* the `.data` of the learnable `scale` and `zero_point` parameters based on the observed `min_val` and `max_val`. This is typically done *before* starting QAT to give the parameters a good starting point. During QAT, these parameters are updated by the optimizer.

### 4. Evaluation Mode (Post-PTQ or Post-QAT)

*   **Configuration**:
    *   `model.eval()` is active.
    *   `qat_mode` and `learnable_params` settings from the preparation phase persist but primary control is via `model.eval()`.
*   **Behavior**:
    *   **Apply Quantization**: If `enabled = True` and `is_calibrated = True`:
        *   It uses its current `scale` and `zero_point` (which are either fixed buffers from PTQ/QAT-fixed, or the final values of `nn.Parameter`s from QAT-learnable) to perform simulated FP4 quantization (quantize then dequantize) of the input tensor `x`.
        *   Standard `torch.round()` is used for quantization if `learnable_params` was false; if `learnable_params` was true, `quantize_to_fp4_asymmetric` is still used (which itself uses `torch.round`) because gradients are not needed in eval mode.
    *   **Pass-Through**: If `enabled = False` or `is_calibrated = False`, it passes the input tensor `x` through unmodified.

## Usage Summary & Best Practices

*   **PTQ**: Use `prepare_model_for_ptq` (sets `qat_mode=False`, `learnable_params=False`). Run `calibrate_model` (which calls `compute_quant_params()` on stubs). Then use `convert_ptq_model_to_eval`.
*   **QAT (Fixed/Batch Stats)**: Use `prepare_model_for_qat` with `learnable_qat_params=False`. During the training loop, stubs will observe and fake quantize. Call `compute_quant_params()` on stubs (while model is in `train()` mode) after QAT to set final scale/zp based on overall stats. Then use `convert_qat_model_to_eval`.
*   **QAT (Learnable Params)**: Use `prepare_model_for_qat` with `learnable_qat_params=True`. Optionally, run a brief calibration phase and call `compute_quant_params()` on stubs (model in `train()` mode) *before* starting QAT to initialize learnable parameters. During QAT, these parameters will be updated by the optimizer. After QAT, directly use `convert_qat_model_to_eval`.

Understanding these modes is key to effectively applying FP4 simulation using IgniteFP4. 