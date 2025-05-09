# Makes ignitefp4_lib a package 

"""
IgniteFP4: A PyTorch-based toolkit for simulating 4-bit floating-point precision in neural networks.

This library provides tools for experimenting with FP4 precision for both model weights and activations.
It supports both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) workflows.

Main Components:
---------------
- numerics: Core functions for FP4 quantization and dequantization
- layers: FP4-simulating versions of standard PyTorch layers (Linear, Conv2d, BatchNorm2d, etc.)
- quantization_utils: High-level functions for preparing, calibrating, and converting models

Example Usage:
-------------
```python
# Post-Training Quantization (PTQ)
from ignitefp4_lib.quantization_utils import prepare_model_for_ptq, calibrate_model, convert_ptq_model_to_eval

# Prepare model for PTQ
ptq_model = prepare_model_for_ptq(float_model)

# Calibrate with representative data
calibrate_model(ptq_model, calibration_loader, device='cuda')

# Convert to evaluation mode for FP4-simulated inference
eval_model = convert_ptq_model_to_eval(ptq_model)

# Quantization-Aware Training (QAT)
from ignitefp4_lib.quantization_utils import prepare_model_for_qat, convert_qat_model_to_eval

# Prepare model for QAT (optionally with learnable quantization parameters)
qat_model = prepare_model_for_qat(float_model, learnable_qat_params=True)

# Train the QAT model (standard PyTorch training loop)
# ...

# Convert to evaluation mode for FP4-simulated inference
eval_model = convert_qat_model_to_eval(qat_model)
```

For more information, see the examples/ directory and documentation.
"""

# Import main components for convenient access
from .numerics import (
    quantize_to_fp4_symmetric, dequantize_from_fp4_symmetric,
    quantize_to_fp4_asymmetric, dequantize_from_fp4_asymmetric,
    quantize_to_fp4_asymmetric_ste,
    calculate_asymmetric_scale_zeropoint,
    FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL,
    FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL
)

from .layers import (
    FP4Linear, FP4Conv2d, FP4BatchNorm2d,
    FP4QuantStub, FP4DequantStub,
    FP4AvgPool2d, FP4MaxPool2d,
    FP4ConvBnReLU2d
)

from .quantization_utils import (
    prepare_model_for_ptq, calibrate_model, convert_ptq_model_to_eval,
    prepare_model_for_qat, convert_qat_model_to_eval
)

# Expose version
__version__ = '0.1.0' 