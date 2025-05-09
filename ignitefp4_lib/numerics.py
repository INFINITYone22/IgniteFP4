# ignitefp4_lib/numerics.py
"""
Core numerical functions for FP4 quantization, supporting both symmetric and asymmetric schemes.
This module provides the low-level operations for calculating quantization parameters (scale and zero-point)
and for converting tensors between floating-point and simulated FP4 representations.

Key functionalities include:
- Symmetric quantization/dequantization (typically for weights).
- Asymmetric quantization/dequantization (typically for activations).
- Straight-Through Estimator (STE) for rounding, enabling gradient flow for learnable QAT parameters.
"""
import torch

# Helper function for Straight-Through Estimator for rounding
def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Applies rounding in the forward pass and estimates gradients as identity in the backward pass.
    
    The standard `torch.round` function has zero gradients almost everywhere, which prevents
    gradient-based optimization of quantization parameters (like scale and zero-point)
    during Quantization-Aware Training (QAT). The Straight-Through Estimator (STE)
    is a common technique to overcome this. In the forward pass, it performs the desired
    discretization (rounding in this case). In the backward pass, it "pretends" that
    the operation was an identity function, allowing gradients to flow through unchanged.

    Args:
        x (torch.Tensor): The input tensor, typically `float_tensor / scale` before rounding.

    Returns:
        torch.Tensor: The tensor with values rounded to the nearest integer.
                      During backpropagation, the gradient of this operation is treated as 1.
    """
    return (x.round() - x).detach() + x

# --- Signed Symmetric Quantization (e.g., for Weights) ---

# Defines the quantization range for signed 4-bit integers.
# For N bits signed, the typical range is -(2^(N-1)) to (2^(N-1) - 1).
# For 4 bits, this translates to -8 to +7, representing 16 distinct levels.
FP4_SIGNED_SYMMETRIC_MIN_VAL = -8
FP4_SIGNED_SYMMETRIC_MAX_VAL = 7

def calculate_symmetric_scale(tensor: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Calculates the per-tensor scale factor for signed symmetric quantization.

    In symmetric quantization, the real value 0.0 is mapped to the quantized value 0.
    The scale factor `s` is determined by the maximum absolute value in the input tensor `r_max_abs`
    and the maximum representable value in the quantized domain `q_max`.
    The formula is `s = r_max_abs / q_max`.
    For signed N-bit quantization, `q_max` is typically `2^(N-1) - 1`.

    Example for 4-bit signed:
    - Quantized range: [-8, 7]
    - `q_max` = 7
    If `tensor.abs().max()` is 2.5, then `scale = 2.5 / 7`.

    Args:
        tensor (torch.Tensor): The input float tensor (e.g., weights) from which to calculate the scale.
        bits (int): The number of bits for quantization. Defaults to 4.
                    Currently, only 4-bit is actively supported by this function's logic.

    Returns:
        torch.Tensor: A scalar tensor representing the calculated scale factor.
                      - Returns 1.0 if the tensor contains only zeros (to avoid 0/0 and ensure 0 maps to 0).
                      - Returns a small positive epsilon if the calculated scale would be zero but `max_abs` is not
                        (e.g., if `max_abs` is extremely small), to prevent division by zero later.
    """
    if bits != 4:
        # This check ensures that the hardcoded quantization range limits are appropriate.
        raise NotImplementedError("Currently only 4-bit symmetric quantization is implemented for scale calculation.")

    # Find the maximum absolute value in the tensor. This will be mapped to the edge of the positive quantized range.
    max_abs = torch.max(torch.abs(tensor))

    # Handle the case where the tensor is all zeros.
    # If max_abs is 0, all elements are 0. Quantized value should be 0.
    # (0 / scale) = 0. Any positive scale works. Scale = 1.0 is a safe choice.
    if max_abs == 0:
        return torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)

    # For signed N-bit quantization, the maximum positive quantized value is 2^(N-1) - 1.
    # For bits=4, this is 2^(4-1) - 1 = 2^3 - 1 = 8 - 1 = 7.
    quantization_range_positive_side = (2**(bits - 1)) - 1
    
    # Calculate the scale: scale = real_max_abs / quantized_max
    scale = max_abs / quantization_range_positive_side
    
    # Handle cases where scale might become zero due to floating point underflow,
    # even if max_abs was not zero.
    if scale == 0:
        # Return a very small positive number (epsilon) to prevent division by zero during quantization.
        # This can happen if max_abs is extremely small but non-zero.
        return torch.tensor(torch.finfo(tensor.dtype).eps, device=tensor.device, dtype=tensor.dtype)
        
    return scale

def quantize_to_fp4_symmetric(float_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a float tensor to a signed 4-bit symmetric representation (simulated).

    The symmetric quantization process is:
    1. Calculate a per-tensor symmetric `scale` factor based on the input tensor's absolute maximum.
       (Zero-point is implicitly 0 for symmetric quantization).
    2. Divide the input `float_tensor` by this `scale`.
    3. Round the result to the nearest integer. This step maps continuous values to discrete integer levels.
    4. Clamp the rounded values to the valid 4-bit signed integer range, which is [-8, 7].
       This ensures that the quantized values fit within the target bit representation.

    The quantized values are stored in an `int8` tensor for convenience and because
    PyTorch does not have a native 4-bit integer type.

    Args:
        float_tensor (torch.Tensor): The input float tensor to be quantized (e.g., model weights).
                                     Must be of a float dtype (float32, float16, bfloat16).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - quantized_tensor (torch.Tensor): The simulated 4-bit quantized values, stored as `torch.int8`.
                                               Values will be in the range [FP4_SIGNED_SYMMETRIC_MIN_VAL, 
                                                                          FP4_SIGNED_SYMMETRIC_MAX_VAL].
            - scale (torch.Tensor): The scalar per-tensor scale factor that was calculated and used.
    """
    if float_tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(f"Input tensor must be a float type, got {float_tensor.dtype}")

    # Step 1: Calculate the symmetric scale for the entire tensor.
    scale = calculate_symmetric_scale(float_tensor, bits=4)
    
    # Step 2: Divide by scale (inverse of X_f = X_q * scale)
    scaled_tensor = float_tensor / scale
    
    # Step 3: Round to the nearest integer.
    # This is the core discretization step.
    rounded_tensor = torch.round(scaled_tensor)
    
    # Step 4: Clamp to the 4-bit signed range [-8, 7].
    # This ensures the values are within the representable range of our target FP4 format.
    quantized_tensor = torch.clamp(rounded_tensor, 
                                   FP4_SIGNED_SYMMETRIC_MIN_VAL, 
                                   FP4_SIGNED_SYMMETRIC_MAX_VAL)
    
    # Store as int8 as there's no native 4-bit type.
    return quantized_tensor.to(torch.int8), scale

def dequantize_from_fp4_symmetric(quantized_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes a signed 4-bit symmetric tensor (simulated as int8) back to a float tensor.

    The dequantization process is straightforward for symmetric quantization:
      `float_value = quantized_value * scale`
    
    This function assumes the `quantized_tensor` contains values that were originally
    quantized using a symmetric scheme and are within the [-8, 7] range.

    Args:
        quantized_tensor (torch.Tensor): The `int8` tensor containing simulated 4-bit quantized values.
                                       These values are expected to be in the range [-8, 7].
        scale (torch.Tensor): The scalar per-tensor scale factor that was used during the
                              corresponding quantization step. Must be a positive value.

    Returns:
        torch.Tensor: The dequantized float tensor. Its dtype will match the dtype of the `scale` tensor.
    """
    if quantized_tensor.dtype != torch.int8:
        raise ValueError("Input quantized_tensor should be int8 (representing 4-bit data).")
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
        raise ValueError("Scale must be a scalar torch.Tensor.")
    
    # A non-positive scale is problematic for dequantization as it can flip signs or yield zeros unexpectedly.
    if scale.item() <= 0: 
        # Using .item() is appropriate for a scalar tensor to get its Python number value.
        print(f"Warning: Dequantizing with non-positive scale: {scale.item()}. "
              "This might indicate an issue with the quantization process or scale calculation.")

    # Convert quantized tensor to the same float type as the scale first, then multiply.
    # This ensures the multiplication is done in floating point.
    float_tensor = quantized_tensor.to(scale.dtype) * scale 
    return float_tensor

# --- Unsigned Asymmetric Quantization (e.g., for Activations) ---

# Defines the quantization range for unsigned 4-bit integers.
# For N bits unsigned, the range is 0 to (2^N - 1).
# For 4 bits, this translates to 0 to 15, representing 16 distinct levels.
FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL = 0
FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL = 15

def calculate_asymmetric_scale_zeropoint(tensor: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates per-tensor scale and zero-point for unsigned asymmetric quantization.

    Asymmetric quantization maps a real-valued range `[r_min, r_max]` from the input tensor
    to a quantized integer range `[q_min, q_max]`. For N-bit unsigned, this is `[0, 2^N - 1]`.
    The relationship is: `real_value = (quantized_value - zero_point) * scale`.

    Derivation:
    1. `r_max = (q_max - zero_point) * scale`
    2. `r_min = (q_min - zero_point) * scale`
    Subtracting (2) from (1):
       `r_max - r_min = (q_max - q_min) * scale`
       `scale = (r_max - r_min) / (q_max - q_min)`
    From (2), `r_min / scale = q_min - zero_point`
       `zero_point = q_min - r_min / scale`
    The zero-point is then rounded to the nearest integer and clamped to `[q_min, q_max]`.

    Example for 4-bit unsigned:
    - Quantized range: `[0, 15]` (`q_min=0`, `q_max=15`)
    If `tensor.min()` is 0.5 and `tensor.max()` is 3.5:
    - `scale = (3.5 - 0.5) / (15 - 0) = 3.0 / 15 = 0.2`
    - `zero_point_float = 0 - 0.5 / 0.2 = -2.5`
    - `zero_point_rounded = round(-2.5) = -2` (or -3 depending on rounding mode, torch.round rounds to nearest even for .5)
    - `zero_point_clamped = clamp(zero_point_rounded, 0, 15)` (e.g., 0 if -2 or -3)

    Args:
        tensor (torch.Tensor): The input float tensor (e.g., a batch of activations).
        bits (int): The number of bits for quantization. Defaults to 4.
                    Currently, only 4-bit is actively supported by this function's logic.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - scale (torch.Tensor): The scalar per-tensor scale factor.
                                    Guaranteed to be a positive value (or a small epsilon if r_min ~ r_max).
            - zero_point (torch.Tensor): The scalar per-tensor zero-point, as `torch.int32`.
                                         It is rounded and clamped to the quantized range `[q_min, q_max]`.
    """
    if bits != 4:
        raise NotImplementedError("Currently only 4-bit asymmetric quantization is implemented.")

    # Determine the actual min and max values in the input tensor.
    # Convert to float for calculations, in case the input tensor is of a lower precision float type.
    min_val = torch.min(tensor).float() 
    max_val = torch.max(tensor).float()

    # Define the quantization range for unsigned N-bit integers.
    # For bits=4, qmin=0, qmax=15.
    qmin = float(FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL) # Typically 0
    qmax = float(FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL) # Typically 2^N - 1

    # Handle the edge case where all values in the tensor are identical.
    if min_val == max_val:
        # If all values are zero.
        if min_val == 0:
            # Real range is 0. Quantized value 0 should map to real 0.
            # (0 - zp) * scale = 0. Choose zp=0, scale=1.0 (any positive scale works).
            return torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype), \
                   torch.tensor(0, device=tensor.device, dtype=torch.int32)
        else:
            # If all values are a non-zero constant (e.g., all 5.0).
            # The range (max_val - min_val) is 0, so scale would be 0 or NaN.
            # To ensure the constant value can be represented, we can:
            # 1. Set scale to a very small number (epsilon) to avoid division by zero.
            # 2. Try to map this constant value to a specific point in the quantized range,
            #    e.g., the middle (qmin + qmax) / 2.
            #    real_val = (quant_val - zp) * eps
            #    zp = quant_val - real_val / eps. Since eps is tiny, real_val/eps is large.
            #    A simpler approach is to set zp such that the real value maps to a quantized value.
            #    Let's set zp so min_val maps to qmin.
            #    zp_float = qmin - min_val / eps.
            #    However, PyTorch's default observers map the single real value to zero_point=0 if scale becomes undefined
            #    or map the value to a quantized value (e.g. qmin) and set scale to effectively 1.
            #    For simplicity, if range is zero, we make scale a tiny positive number and aim
            #    to place the single value such that quant(value) = zero_point, resulting in 0 after (x_q - z_p).
            #    This isn't ideal. A better way for constant tensor:
            #    scale = 1.0 (or some default, or based on magnitude if not 0)
            #    zero_point = round(q_min - min_val / scale_effective) -> needs careful thought.
            #
            #    A common practice: if min_val == max_val, make scale = 1 (or eps if min_val is huge)
            #    and adjust zero_point to map min_val to the center of the quant range if possible, or qmin.
            #    If min_val = max_val, set scale to a small value (epsilon) and set zero_point such that
            #    the quantized value of min_val is qmin.
            #    quantized_value = round(min_val / scale + zero_point_raw)
            #    qmin = min_val / scale + zero_point_raw  => zero_point_raw = qmin - min_val / scale
            zero_point_float = qmin - (min_val / scale) # This will be large
            # Clamp to ensure it's within [qmin, qmax] after rounding.
            # This ensures that dequant(qmin) with this zp and scale is close to min_val.
            # (qmin - (qmin - min_val/scale)) * scale = min_val.
            # This is the standard formula, clamping will handle extreme cases.
            zero_point = torch.round(zero_point_float)
            zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int32)
            return torch.tensor(scale, device=tensor.device, dtype=tensor.dtype), zero_point

    # Calculate scale: s = (r_max - r_min) / (q_max - q_min)
    # This ensures the full range of real values maps to the full range of quantized values.
    scale = (max_val - min_val) / (qmax - qmin)

    # Handle case where scale might be zero (e.g., if max_val and min_val are extremely close but not identical,
    # leading to (max_val - min_val) underflowing to 0).
    # A scale of 0 would lead to division by zero during quantization or dequantization.
    if scale == 0:
        scale = torch.tensor(torch.finfo(tensor.dtype).eps, device=tensor.device, dtype=tensor.dtype)

    # Calculate zero-point: z = q_min - r_min / s
    # This formula ensures that `r_min` correctly maps to `q_min` after quantization and rounding.
    # Proof:
    #   quantized_value_for_r_min = round(r_min / scale + zero_point_float)
    #                             = round(r_min / scale + (q_min - r_min / scale))
    #                             = round(q_min) = q_min
    zero_point_float = qmin - (min_val / scale)
    
    # Round zero_point to the nearest integer. This is important because zero_point itself is an integer concept.
    zero_point = torch.round(zero_point_float)
    
    # Clamp the zero_point to be within the quantization range [q_min, q_max].
    # This is a standard practice, as specified in many quantization schemes (e.g., TFLite).
    # It ensures the zero_point itself is a valid quantized value.
    zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int32)
        
    return torch.tensor(scale, device=tensor.device, dtype=tensor.dtype), zero_point

def quantize_to_fp4_asymmetric(float_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    Quantizes a float tensor to an unsigned 4-bit asymmetric representation (simulated)
    using pre-calculated scale and zero-point.

    The asymmetric quantization formula is:
      `quantized_value = round(float_value / scale + zero_point)`

    The result is then clamped to the 4-bit unsigned integer range [0, 15].
    The quantized values are stored in an `int8` tensor for convenience.

    Args:
        float_tensor (torch.Tensor): The input float tensor to be quantized (e.g., activations).
        scale (torch.Tensor): The scalar per-tensor scale factor. Must be positive.
        zero_point (torch.Tensor): The scalar per-tensor zero-point. Expected to be an integer
                                   within the target quantized range (e.g., 0-15 for 4-bit).

    Returns:
        torch.Tensor: The simulated 4-bit quantized values, stored as `torch.int8`.
                      Values will be in the range [FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL,
                                                   FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL].
    """
    if float_tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(f"Input tensor must be a float type, got {float_tensor.dtype}")
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1 or scale.item() <= 0:
        raise ValueError("Scale must be a positive scalar torch.Tensor.")
    if not isinstance(zero_point, torch.Tensor) or zero_point.numel() != 1:
        raise ValueError("Zero-point must be a scalar torch.Tensor.")
    
    # Formula: X_q = round(X_f / scale + ZP)
    # Convert zero_point to the float tensor's dtype for the addition.
    quantized_val_float = (float_tensor / scale) + zero_point.to(float_tensor.dtype)
    
    # Round to the nearest integer to get the discrete quantized levels.
    rounded_quantized_val = torch.round(quantized_val_float)
    
    # Clamp to the unsigned 4-bit range [0, 15].
    clamped_quantized_val = torch.clamp(rounded_quantized_val, 
                                        FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL, 
                                        FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL)
                                        
    return clamped_quantized_val.to(torch.int8)

def quantize_to_fp4_asymmetric_ste(float_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    Quantizes a float tensor to an unsigned 4-bit asymmetric representation (simulated)
    using pre-calculated scale and zero-point, with Straight-Through Estimator (STE) for rounding.

    This function is intended for Quantization-Aware Training (QAT) where `scale` and/or
    `zero_point` might be learnable parameters. The STE allows gradients to flow
    back through the rounding operation.

    The process:
    1. `scaled_input = float_tensor / scale`
    2. `shifted_input = scaled_input + zero_point`
    3. `rounded_output = _ste_round(shifted_input)` (rounding with STE)
    4. `clamped_output = clamp(rounded_output, q_min, q_max)`

    The quantized values are stored in an `int8` tensor.

    Args:
        float_tensor (torch.Tensor): The input float tensor (e.g., activations during QAT).
        scale (torch.Tensor): The scalar per-tensor scale factor. Expected to be learnable or fixed.
        zero_point (torch.Tensor): The scalar per-tensor zero-point. Expected to be learnable or fixed.
                                   Should be treated as a float during forward pass if learnable,
                                   though its conceptual role is integer-like.

    Returns:
        torch.Tensor: The simulated 4-bit quantized values with STE applied to rounding, stored as `torch.int8`.
                      Values will be in the range [FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL,
                                                   FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL].
    """
    if float_tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(f"Input tensor must be a float type, got {float_tensor.dtype}")
    # Scale can be learnable, so we don't strictly enforce positive here, but usually it should be.
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
        raise ValueError("Scale must be a scalar torch.Tensor.")
    if not isinstance(zero_point, torch.Tensor) or zero_point.numel() != 1:
        raise ValueError("Zero-point must be a scalar torch.Tensor.")

    # X_q = round(X_f / scale + ZP)
    # Perform calculations in the dtype of float_tensor to maintain precision until rounding.
    # Ensure zero_point is of the same dtype as (float_tensor / scale) for the addition.
    val_for_rounding = (float_tensor / scale) + zero_point.to(float_tensor.dtype)
    
    # Apply rounding using Straight-Through Estimator.
    # This is the key difference for QAT, allowing gradients to pass through the rounding step.
    rounded_val = _ste_round(val_for_rounding)
    
    # Clamp to the unsigned 4-bit range [0, 15].
    # This ensures the output conforms to the target bit representation.
    clamped_val = torch.clamp(rounded_val, 
                              FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL, 
                              FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL)
                              
    return clamped_val.to(torch.int8)

def dequantize_from_fp4_asymmetric(quantized_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes an unsigned 4-bit asymmetric tensor (simulated as int8) back to float.

    The dequantization formula is:
      `float_value = (quantized_value - zero_point) * scale`

    This function assumes `quantized_tensor` contains values originally quantized
    using an asymmetric scheme and are within the [0, 15] range.

    Args:
        quantized_tensor (torch.Tensor): The `int8` tensor containing simulated 4-bit quantized values
                                       (expected to be in the range [0, 15]).
        scale (torch.Tensor): The scalar per-tensor scale factor used during quantization. Must be positive.
        zero_point (torch.Tensor): The scalar per-tensor zero-point used during quantization.
                                   Expected to be an integer within the target quantized range.

    Returns:
        torch.Tensor: The dequantized float tensor. Its dtype will match the dtype of the `scale` tensor.
    """
    if quantized_tensor.dtype != torch.int8:
        raise ValueError("Input quantized_tensor should be int8 (representing 4-bit data).")
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1 or scale.item() <= 0:
        raise ValueError("Scale must be a positive scalar torch.Tensor.")
    if not isinstance(zero_point, torch.Tensor) or zero_point.numel() != 1:
        raise ValueError("Zero-point must be a scalar torch.Tensor.")

    # Convert quantized_tensor and zero_point to the float dtype of the scale tensor before subtraction.
    # This ensures the subtraction and multiplication are done in floating-point.
    # X_f = (X_q - ZP) * scale
    float_tensor = (quantized_tensor.to(scale.dtype) - zero_point.to(scale.dtype)) * scale
    return float_tensor 