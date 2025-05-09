# examples/simple_ptq_example.py
"""
Example script demonstrating the Post-Training Quantization (PTQ) workflow
using the IgniteFP4 library.

This script covers the following steps:
1.  **Model Definition**: Defines a simple CNN (`SimpleExampleModel`) for image classification.
2.  **Float Model Initialization**: Creates an instance of the float32 model.
3.  **PTQ Preparation**: Uses `ignitefp4_lib.quantization_utils.prepare_model_for_ptq`
    to convert the float32 model into an FP4-simulated model. This involves:
    - Replacing `nn.Conv2d` and `nn.Linear` layers with `FP4Conv2d` and `FP4Linear`.
    - Inserting `FP4QuantStub` modules before these FP4 layers to handle activation
      quantization.
4.  **Calibration Data**: Generates dummy data to be used for calibrating the quantization
    parameters of the `FP4QuantStub`s.
5.  **Model Calibration**: Uses `ignitefp4_lib.quantization_utils.calibrate_model`
    to feed the calibration data through the prepared model. During this step,
    `FP4QuantStub`s (in PTQ mode) observe the range of activation values and then
    compute their `scale` and `zero_point` quantization parameters.
6.  **Conversion to Eval Mode**: Uses `ignitefp4_lib.quantization_utils.convert_ptq_model_to_eval`
    to set the calibrated model to evaluation mode. In this mode, `FP4QuantStub`s
    will use their computed parameters to perform fake quantization.
7.  **Inference**: Performs a sample inference pass with the PTQ model.
8.  **Comparison (Optional)**: Compares the output of the PTQ model with the original
    float32 model to give an idea of the quantization effect.

This example uses dummy data and a simple model for clarity. In a real-world scenario,
actual pre-trained model weights and a representative calibration dataset would be used.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Adjust path to import from ignitefp4_lib
# This is for running the example directly from the examples directory
# For library usage, you'd typically install ignitefp4_lib or have it in PYTHONPATH
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ignitefp4_lib.layers import FP4Linear, FP4Conv2d, FP4QuantStub
    from ignitefp4_lib.quantization_utils import prepare_model_for_ptq, calibrate_model, convert_ptq_model_to_eval
except ImportError as e:
    print(f"Error importing ignitefp4_lib: {e}")
    print("Please ensure ignitefp4_lib is installed or the PYTHONPATH is set correctly.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Define a Simple Model (copied from tests for self-containment) ---
class SimpleExampleModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 7 * 7, num_classes) # Assuming 28x28 input, 8*7*7=392

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# --- Helper to get dummy data (copied from tests) ---
def get_dummy_dataloader(batch_size=4, num_batches=10, in_channels=1, img_size=28, device='cpu'):
    data = []
    for _ in range(num_batches * batch_size): # total samples
        data.append(torch.randn(in_channels, img_size, img_size)) # individual samples
    
    # Create a dataset of individual samples
    full_dataset = TensorDataset(torch.stack(data))
    # DataLoader handles batching
    return DataLoader(full_dataset, batch_size=batch_size)

def run_ptq_example():
    print("Starting IgniteFP4 Simple PTQ Workflow Example...")
    
    # --- Configuration ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    img_size = 28 # For SimpleExampleModel with 28x28 input
    in_channels = 1
    num_classes = 10
    
    # --- 1. Create/Load a Float32 Model ---
    # In a real scenario, you would load your pre-trained float32 model here.
    print("\n--- Step 1: Initialize Float32 Model ---")
    float_model = SimpleExampleModel(in_channels=in_channels, num_classes=num_classes).to(device)
    float_model.eval() # Start with the model in evaluation mode, as if it were pre-trained.
    print("Float32 model created:")
    print(float_model)

    # --- 2. Prepare Model for PTQ ---
    # `prepare_model_for_ptq` replaces supported nn.Modules with their FP4 counterparts
    # (e.g., nn.Linear -> FP4Linear) and inserts FP4QuantStubs before them.
    # These stubs will be used to observe activation ranges during calibration.
    # By default, this creates a deepcopy of the model.
    print("\n--- Step 2: Prepare Model for PTQ ---")
    ptq_prepared_model = prepare_model_for_ptq(float_model)
    ptq_prepared_model.to(device)
    print("Model prepared for PTQ:")
    print(ptq_prepared_model)
    
    # Optional: Verify that specific layers have been replaced as expected.
    # For example, `conv1` should now be a nn.Sequential(FP4QuantStub, FP4Conv2d).
    if hasattr(ptq_prepared_model, 'conv1') and isinstance(ptq_prepared_model.conv1, nn.Sequential):
        print(f"  ptq_prepared_model.conv1 is nn.Sequential: {isinstance(ptq_prepared_model.conv1[0], FP4QuantStub)} and {isinstance(ptq_prepared_model.conv1[1], FP4Conv2d)}")
    if hasattr(ptq_prepared_model, 'fc1') and isinstance(ptq_prepared_model.fc1, nn.Sequential):
        print(f"  ptq_prepared_model.fc1 is nn.Sequential: {isinstance(ptq_prepared_model.fc1[0], FP4QuantStub)} and {isinstance(ptq_prepared_model.fc1[1], FP4Linear)}")

    # --- 3. Create Calibration Dataloader ---
    # For PTQ, a small, representative dataset is needed to calibrate the quantization
    # parameters for activations. Here, we use dummy random data.
    print("\n--- Step 3: Create Calibration Dataloader ---")
    calibration_loader = get_dummy_dataloader(batch_size=8, num_batches=5, 
                                              in_channels=in_channels, img_size=img_size, device=device)
    print(f"Calibration dataloader created with {len(calibration_loader.dataset)} samples in {len(calibration_loader)} batches.")

    # --- 4. Calibrate the Model ---
    # `calibrate_model` feeds the calibration data through the prepared model.
    # During this process:
    #   - The model is set to `train()` mode temporarily. This allows `FP4QuantStub`s
    #     (which are in PTQ mode: qat_mode=False) to observe input statistics (min/max).
    #   - After all calibration data is processed, `compute_quant_params()` is called
    #     on each `FP4QuantStub` to calculate and store its scale and zero-point.
    print("\n--- Step 4: Calibrate the Model ---")
    calibrate_model(ptq_prepared_model, calibration_loader, device=device)
    print("Model calibration finished.")
    
    # Optional: Check if the stubs have been calibrated.
    calibrated_stubs_count = 0
    total_stubs_count = 0
    for name, module in ptq_prepared_model.named_modules():
        if isinstance(module, FP4QuantStub):
            total_stubs_count+=1
            if module.is_calibrated:
                calibrated_stubs_count+=1
                # print(f"  Stub '{name}' is calibrated: scale={module.scale.item():.4f}, zp={module.zero_point.item()}")
            else:
                print(f"  Warning: Stub '{name}' was NOT calibrated.")
    print(f"  {calibrated_stubs_count}/{total_stubs_count} stubs are calibrated.")


    # --- 5. Convert Model to Evaluation Mode ---
    # `convert_ptq_model_to_eval` simply calls `model.eval()`.
    # This is crucial because in `eval()` mode:
    #   - `FP4QuantStub`s will use their calibrated parameters to perform fake quantization
    #     (quantize then dequantize) on their inputs.
    #   - Layers like BatchNorm use their running statistics.
    #   - Dropout layers are disabled.
    print("\n--- Step 5: Convert Calibrated Model to Evaluation Mode ---")
    ptq_eval_model = convert_ptq_model_to_eval(ptq_prepared_model)
    print("Model converted to evaluation mode for inference.")
    print(f"  ptq_eval_model.training = {ptq_eval_model.training}")
    # Check a specific stub's mode to confirm it's also in eval mode.
    if hasattr(ptq_eval_model, 'conv1') and isinstance(ptq_eval_model.conv1, nn.Sequential) and isinstance(ptq_eval_model.conv1[0], FP4QuantStub):
        print(f"  ptq_eval_model.conv1[0].training (QuantStub) = {ptq_eval_model.conv1[0].training}")


    # --- 6. Perform Inference with the PTQ Model ---
    # Now the `ptq_eval_model` will simulate FP4 arithmetic internally for weights and activations.
    print("\n--- Step 6: Perform Inference with PTQ Model ---")
    dummy_input_for_inference = torch.randn(1, in_channels, img_size, img_size, device=device)
    with torch.no_grad(): # Inference does not require gradient tracking.
        output = ptq_eval_model(dummy_input_for_inference)
    print(f"Output shape from PTQ model: {output.shape}")
    print(f"Output sample from PTQ model:\n{output}")

    # --- Compare with Float Model (Optional) ---
    # This step helps to understand the impact of PTQ on model accuracy.
    # The difference is expected due to the precision loss from quantization.
    print("\n--- Optional: Compare with Float Model Output ---")
    with torch.no_grad():
        output_float = float_model(dummy_input_for_inference) # Ensure float_model is on the same device.
    print(f"Output shape from Float model: {output_float.shape}")
    print(f"Output sample from Float model:\n{output_float}")
    
    if output.shape == output_float.shape:
        abs_diff = torch.abs(output - output_float).mean()
        print(f"Mean absolute difference between PTQ and Float model outputs: {abs_diff.item():.4f}")
    else:
        print("Output shapes differ, cannot compute difference.")

    print("\nIgniteFP4 Simple PTQ Workflow Example Finished.")

if __name__ == '__main__':
    run_ptq_example() 