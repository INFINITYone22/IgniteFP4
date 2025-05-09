# IgniteFP4: A PyTorch Framework for FP4 Precision Neural Networks

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

IgniteFP4 is a comprehensive PyTorch-based framework for exploring 4-bit floating-point (FP4) precision in deep learning models. It provides numerics simulation, custom layers, and robust quantization workflows for both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

Built with researchers and ML engineers in mind, IgniteFP4 makes it easy to experiment with ultra-low precision neural networks while maintaining good accuracy. The framework is designed to be forward-compatible with upcoming hardware that may support native FP4 operations.

## 🔥 Key Features

*   **FP4 Numerics Simulation**:
    *   Signed symmetric FP4 quantization for weights
    *   Unsigned asymmetric FP4 quantization for activations
    *   Straight-Through Estimator (STE) for gradient-based optimization
    
*   **Custom PyTorch Layers**:
    *   `FP4Linear`, `FP4Conv2d`, `FP4BatchNorm2d` - Core layers with FP4-simulated weights
    *   `FP4QuantStub` - For activation quantization with calibration and QAT support
    *   `FP4ConvBnReLU2d` - Fused layer for common CNN operations
    
*   **Complete Quantization Workflows**:
    *   Post-Training Quantization (PTQ) with calibration
    *   Quantization-Aware Training (QAT) with optional learnable parameters
    *   Simple model conversion utilities

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/INFINITYone22/IgniteFP4.git
cd IgniteFP4

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Post-Training Quantization (PTQ)

```python
from ignitefp4_lib.quantization_utils import prepare_model_for_ptq, calibrate_model, convert_ptq_model_to_eval
import torch

# 1. Load your pre-trained model
model = YourModel()
model.load_state_dict(torch.load('your_model_weights.pth'))

# 2. Prepare model for PTQ
ptq_model = prepare_model_for_ptq(model)

# 3. Calibrate with representative data
calibrate_model(ptq_model, calibration_loader, device='cuda')

# 4. Convert to evaluation mode
eval_model = convert_ptq_model_to_eval(ptq_model)

# 5. Run inference with simulated FP4 precision
with torch.no_grad():
    outputs = eval_model(inputs)
```

### Quantization-Aware Training (QAT)

```python
from ignitefp4_lib.quantization_utils import prepare_model_for_qat, convert_qat_model_to_eval
import torch

# 1. Load your pre-trained model
model = YourModel()
model.load_state_dict(torch.load('your_model_weights.pth'))

# 2. Prepare model for QAT with learnable quantization parameters
qat_model = prepare_model_for_qat(model, learnable_qat_params=True)
qat_model.to('cuda')

# 3. Fine-tune the model
optimizer = torch.optim.Adam(qat_model.parameters())
criterion = torch.nn.CrossEntropyLoss()

qat_model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        
        optimizer.zero_grad()
        outputs = qat_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# 4. Convert to evaluation mode
eval_model = convert_qat_model_to_eval(qat_model)

# 5. Run inference with simulated FP4 precision
with torch.no_grad():
    outputs = eval_model(inputs)
```

## 📚 Documentation

The project is fully documented with comprehensive docstrings, helping you understand every component:

- `ignitefp4_lib/numerics.py`: Core quantization and dequantization functions
- `ignitefp4_lib/layers.py`: FP4-simulated PyTorch layers
- `ignitefp4_lib/quantization_utils.py`: High-level quantization workflows
- `examples/`: End-to-end examples for both PTQ and QAT

For deeper insights, see the `docs/` directory containing additional documentation.

## 🛣️ Roadmap

- Advanced QAT techniques with learnable scale and zero-point initialization
- Support for more PyTorch layers
- Per-channel quantization for improved accuracy
- Integration with upcoming hardware backends
- Pre-quantized model zoo with benchmarks
- PyPI packaging for easier installation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

Copyright (c) 2025 ROHITH GARAPATI

Released under the MIT License. See [LICENSE](LICENSE) for details.

## 📧 Contact

ROHITH GARAPATI - GitHub: [@INFINITYone22](https://github.com/INFINITYone22) 