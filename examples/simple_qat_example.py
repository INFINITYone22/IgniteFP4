# examples/simple_qat_example.py
"""
Example script demonstrating the Quantization-Aware Training (QAT) workflow
using the IgniteFP4 library.

This script covers the following steps:
1.  **Model Definition**: Defines a simple CNN (`SimpleExampleModel`) for image classification.
2.  **Float Model Initialization**: Creates an instance of the float32 model. For QAT,
    this model would typically be pre-trained, but here we start with random weights
    for a self-contained example.
3.  **QAT Preparation**: Uses `ignitefp4_lib.quantization_utils.prepare_model_for_qat`
    to convert the float32 model. This involves:
    - Replacing `nn.Conv2d` and `nn.Linear` with `FP4Conv2d` and `FP4Linear`.
    - Inserting `FP4QuantStub` modules (with `qat_mode=True`) before these FP4 layers.
      In QAT mode, these stubs perform fake quantization during the forward pass in training.
4.  **QAT Fine-tuning**: Simulates a few epochs of training. During this phase:
    - The model is in `train()` mode.
    - `FP4QuantStub`s apply fake quantization (quantize then dequantize) to activations.
    - Gradients flow through the fake quantized graph, allowing the model to adapt to
      the effects of quantization.
    - If `learnable_params=True` was used in `FP4QuantStub` (via `prepare_model_for_qat`),
      quantization parameters themselves would be learned.
5.  **Finalize Parameters & Convert to Eval Mode**: 
    - After QAT, `FP4QuantStub.compute_quant_params()` is called on each stub. 
      The model should be in `train()` mode for this step so stubs can use their 
      accumulated min/max statistics from the QAT process to calculate and set their
      final `scale` and `zero_point` values. This is crucial if not using learnable parameters
      or if a final calibration based on overall training statistics is desired.
    - The model is then converted to evaluation mode using
      `ignitefp4_lib.quantization_utils.convert_qat_model_to_eval` (which calls `model.eval()`).
6.  **Inference**: Performs a sample inference pass with the QAT model.
7.  **Comparison (Optional)**: Compares the QAT model output with the original float model.
    Note: Since the float model is untrained and the QAT model undergoes some training,
    this comparison mainly checks for operational consistency rather than accuracy equivalence
    without a proper training setup.

This example uses dummy data and a simple model for clarity.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Adjust path to import from ignitefp4_lib
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ignitefp4_lib.layers import FP4Linear, FP4Conv2d, FP4QuantStub
    # For QAT, after training, we finalize stub parameters and then convert to eval.
    # convert_qat_model_to_eval is used for clarity, though functionally it just calls .eval().
    from ignitefp4_lib.quantization_utils import prepare_model_for_qat, convert_qat_model_to_eval
except ImportError as e:
    print(f"Error importing ignitefp4_lib: {e}")
    print("Please ensure ignitefp4_lib is installed or the PYTHONPATH is set correctly.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Define a Simple Model (copied from PTQ example for self-containment) ---
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
        self.fc1 = nn.Linear(8 * 7 * 7, num_classes) # Assuming 28x28 input

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

# --- Helper to get dummy data (copied from PTQ example) ---
# Added labels for loss calculation in QAT.
def get_dummy_dataloader(batch_size=4, num_batches=10, in_channels=1, img_size=28, device='cpu', num_classes=10):
    data = []
    labels = []
    for _ in range(num_batches * batch_size):
        data.append(torch.randn(in_channels, img_size, img_size))
        labels.append(torch.randint(0, num_classes, (1,)).item())
    
    full_dataset = TensorDataset(torch.stack(data), torch.tensor(labels, dtype=torch.long))
    return DataLoader(full_dataset, batch_size=batch_size)

def run_qat_example():
    print("Starting IgniteFP4 Simple QAT Workflow Example...")
    
    # --- Configuration ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    img_size = 28
    in_channels = 1
    num_classes = 10
    num_qat_epochs = 3         # Number of epochs for QAT fine-tuning (example only)
    learning_rate = 0.01
    batch_size_qat = 8
    num_qat_batches = 10       # Number of batches per epoch for QAT (example only)
    learn_stub_params = False  # Set to True to test learnable scale/zero-point in FP4QuantStub

    # --- 1. Create/Load a Float32 Model ---
    print("\n--- Step 1: Initialize Float32 Model ---")
    # For a typical QAT workflow, you would start with a pre-trained float32 model.
    # For this example, we initialize a new model with random weights for simplicity.
    float_model = SimpleExampleModel(in_channels=in_channels, num_classes=num_classes).to(device)
    # If loading pre-trained weights, you might do: float_model.load_state_dict(...) and float_model.eval() before QAT prep.
    print("Float32 model created (randomly initialized for this QAT example):")
    # print(float_model) # Can be verbose

    # --- 2. Prepare Model for QAT ---
    # `prepare_model_for_qat` is similar to `prepare_model_for_ptq` but configures
    # `FP4QuantStub`s with `qat_mode=True` (and optionally `learnable_params=True`).
    # This enables fake quantization during the training forward pass.
    print("\n--- Step 2: Prepare Model for QAT ---")
    qat_model = prepare_model_for_qat(float_model, learnable_qat_params=learn_stub_params)
    qat_model.to(device)
    print(f"Model prepared for QAT (learnable stub params: {learn_stub_params}):")
    # print(qat_model) # Can be verbose

    # Verify that stubs are correctly configured for QAT.
    for name, module in qat_model.named_modules():
        if isinstance(module, FP4QuantStub):
            assert module.qat_mode, f"Stub {name} is not in QAT mode!"
            assert module.learnable_params == learn_stub_params, f"Stub {name} learnable_params is not {learn_stub_params}"
    print(f"  All FP4QuantStubs are in qat_mode=True, learnable_params={learn_stub_params}.")

    # --- 3. QAT: Fine-tune the Model ---
    # This part simulates the Quantization-Aware Training process.
    # The model is trained for a few epochs with fake quantization active in FP4QuantStubs.
    # Gradients flow through these stubs (using STE if learnable_params=True and training).
    print(f"\n--- Step 3: Simulate QAT Fine-tuning for {num_qat_epochs} epochs ---")
    qat_model.train() # Set model to training mode for QAT.
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy training data loader (ensure num_classes matches model output)
    train_loader = get_dummy_dataloader(batch_size=batch_size_qat, num_batches=num_qat_batches,
                                        in_channels=in_channels, img_size=img_size, device=device, num_classes=num_classes)

    for epoch in range(num_qat_epochs):
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = qat_model(inputs) # Forward pass with fake quantization
            loss = criterion(outputs, targets)
            loss.backward()             # Gradients flow through the (potentially) quantized graph
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{num_qat_epochs}, Average Loss: {epoch_loss/len(train_loader):.4f}")
    print("Simulated QAT fine-tuning finished.")

    # --- 4. Finalize Quantization Parameters & Convert to Eval Mode ---
    print("\n--- Step 4: Finalize Stub Parameters and Convert to Evaluation Mode ---")
    # After QAT, FP4QuantStubs have observed activation statistics throughout training (or learned them).
    # If not using learnable parameters, or if you want to ensure final parameters are based on overall stats,
    # call `compute_quant_params()` on each stub.
    # The model should be in `train()` mode so stubs use their observed min/max for this computation.
    qat_model.train() 
    print("  Finalizing FP4QuantStub parameters based on observed QAT statistics...")
    for name, module in qat_model.named_modules():
        if isinstance(module, FP4QuantStub):
            if not module.learnable_params:
                 module.compute_quant_params() # Update scale/zp from observed min/max if not learnable
            # For learnable params, they are already trained. compute_quant_params can initialize them
            # before training, or could be called here to log the final learned/observed values.
            # If called on learnable params after training, it would overwrite learned values with batch stats based ones.
            # So, for learnable, we typically rely on the learned values themselves.
            # However, the print statement in compute_quant_params shows the current state.
            if module.learnable_params: # If learnable, ensure they are calibrated (compute_quant_params sets this flag after init)
                if not module.is_calibrated: module.compute_quant_params() # Initialize if not done
            
            final_zp = module.zero_point.item()
            print(f"    Stub '{name}': learnable={module.learnable_params}, calibrated={module.is_calibrated}, scale={module.scale.item():.4f}, zero_point={final_zp:.4f}, obs_min={module.min_val.item():.4f}, obs_max={module.max_val.item():.4f}")

    # Convert the QAT model to evaluation mode for inference.
    # This sets `model.training` to False, which affects BatchNorm, Dropout, and FP4QuantStub behavior
    # (e.g., stubs use finalized params for quantization).
    qat_eval_model = convert_qat_model_to_eval(qat_model)
    print("QAT model finalized and converted to evaluation mode.")
    print(f"  qat_eval_model.training = {qat_eval_model.training}")

    # --- 5. Perform Inference with the QAT Model ---
    # The `qat_eval_model` will now use the finalized/learned quantization parameters for inference.
    print("\n--- Step 5: Perform Inference with QAT Model ---")
    dummy_input_for_inference = torch.randn(1, in_channels, img_size, img_size, device=device)
    with torch.no_grad():
        output_qat = qat_eval_model(dummy_input_for_inference)
    print(f"Output shape from QAT model: {output_qat.shape}")
    print(f"Output sample from QAT model:\n{output_qat}")

    # --- Compare with Original Float Model (Optional) ---
    # Note: The float_model here was not trained, and the QAT model underwent only minimal training.
    # In a real scenario, you'd compare against the *fully trained* float_model (the one QAT started from)
    # and the QAT model after full fine-tuning.
    print("\n--- Optional: Compare with (Untrained) Float Model Output ---")
    float_model.to(device).eval() # Ensure it's on the same device and in eval mode for comparison.
    with torch.no_grad():
        output_float = float_model(dummy_input_for_inference)
    print(f"Output shape from Float model: {output_float.shape}")
    # print(f"Output sample from Float model:\n{output_float}") # Can be verbose
        
    if output_qat.shape == output_float.shape:
        abs_diff = torch.abs(output_qat - output_float).mean()
        # This difference can be large as the float model is untrained and QAT model underwent some training.
        print(f"Mean absolute difference between QAT and (untrained) Float model outputs: {abs_diff.item():.4f}")
    else:
        print("Output shapes differ, cannot compute difference.")

    print("\nIgniteFP4 Simple QAT Workflow Example Finished.")

if __name__ == '__main__':
    run_qat_example() 