# tests/test_quantization_utils.py
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Adjust path to import from ignitefp4_lib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ignitefp4_lib.layers import FP4Linear, FP4Conv2d, FP4BatchNorm2d, FP4QuantStub
from ignitefp4_lib.quantization_utils import prepare_model_for_ptq, calibrate_model, convert_ptq_model_to_eval, prepare_model_for_qat

class SimpleTestModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1) # 28x28 -> 4x28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # 4x28x28 -> 4x14x14
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # 4x14x14 -> 8x14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # 8x14x14 -> 8x7x7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class TestPTQWorkflow(unittest.TestCase):

    def _get_dummy_dataloader(self, batch_size=4, num_batches=5, in_channels=1, img_size=28):
        data = []
        for _ in range(num_batches):
            data.append(torch.randn(batch_size, in_channels, img_size, img_size))
        dataset = TensorDataset(torch.cat(data))
        return DataLoader(dataset, batch_size=batch_size)

    def test_simple_model_ptq_workflow(self):
        print("\nTesting PTQ workflow on a simple CNN model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Create a simple float model
        float_model = SimpleTestModel(in_channels=1, num_classes=10).to(device)
        float_model.eval() # Start with eval mode for consistent initial state if any BN were present

        # 2. Prepare model for PTQ
        print("  Preparing model for PTQ...")
        prepared_model = prepare_model_for_ptq(float_model)
        self.assertIsNotNone(prepared_model)
        # Check some replacements
        self.assertIsInstance(prepared_model.conv1, nn.Sequential) # Conv1 replaced by Seq(Stub, FP4Conv)
        self.assertIsInstance(prepared_model.conv1[0], FP4QuantStub)
        self.assertIsInstance(prepared_model.conv1[1], FP4Conv2d)
        self.assertFalse(prepared_model.conv1[0].qat_mode) # Stub should be in PTQ mode
        self.assertIsInstance(prepared_model.fc1, nn.Sequential) # FC1 replaced by Seq(Stub, FP4Linear)
        self.assertIsInstance(prepared_model.fc1[0], FP4QuantStub)
        self.assertIsInstance(prepared_model.fc1[1], FP4Linear)
        print("  Model preparation: PASSED")

        # 3. Create dummy calibration dataloader
        calib_loader = self._get_dummy_dataloader(batch_size=2, num_batches=3, in_channels=1, img_size=28)

        # 4. Calibrate the model
        print("  Calibrating model...")
        # Ensure model is on the correct device for calibration function to pick up
        prepared_model.to(device) 
        calibrate_model(prepared_model, calib_loader, device=device)
        
        # Check if stubs are calibrated
        calibrated_stubs = 0
        total_stubs = 0
        for module in prepared_model.modules():
            if isinstance(module, FP4QuantStub):
                total_stubs += 1
                if module.is_calibrated:
                    calibrated_stubs += 1
        self.assertGreater(total_stubs, 0, "No FP4QuantStubs found in the prepared model.")
        self.assertEqual(calibrated_stubs, total_stubs, f"Not all stubs were calibrated. Calibrated: {calibrated_stubs}/{total_stubs}")
        self.assertTrue(prepared_model.training, "Model should be in train() mode after calibrate_model returns (as per calibrate_model logic)")
        print("  Model calibration: PASSED")

        # 5. Convert to eval mode
        print("  Converting model to eval mode...")
        eval_model = convert_ptq_model_to_eval(prepared_model)
        self.assertFalse(eval_model.training, "Model should be in eval() mode after conversion.")
        # Check a stub's mode (it should also be in eval mode)
        self.assertFalse(eval_model.conv1[0].training, "Stub should be in eval() mode.")
        print("  Model to eval conversion: PASSED")

        # 6. Perform a forward pass
        print("  Performing forward pass on PTQ model...")
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
        try:
            with torch.no_grad():
                output = eval_model(dummy_input)
            self.assertEqual(output.shape, (1, 10))
            print("  Forward pass on PTQ model: PASSED")
        except Exception as e:
            self.fail(f"Forward pass on PTQ model failed: {e}")
        
        print("PTQ workflow test: COMPLETED")

    def test_simple_model_qat_workflow(self):
        print("\nTesting QAT workflow on a simple CNN model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Create a simple float model
        float_model_qat = SimpleTestModel(in_channels=1, num_classes=10).to(device)
        # For QAT, typically start from pretrained float weights
        float_model_qat.eval() # Or load actual pretrained weights

        # 2. Prepare model for QAT
        print("  Preparing model for QAT...")
        qat_model = prepare_model_for_qat(float_model_qat) # Default inplace=False
        self.assertIsNotNone(qat_model)
        qat_model.to(device)

        # Check stubs are in QAT mode
        stubs_in_qat_mode = 0
        total_stubs = 0
        for module_name, module in qat_model.named_modules():
            if isinstance(module, FP4QuantStub):
                total_stubs += 1
                if module.qat_mode:
                    stubs_in_qat_mode += 1
        self.assertGreater(total_stubs, 0, "No FP4QuantStubs found in QAT model.")
        self.assertEqual(stubs_in_qat_mode, total_stubs, "Not all stubs in QAT model have qat_mode=True.")
        print("  Model preparation for QAT: PASSED")

        # 3. Simulate a brief training loop
        print("  Simulating QAT training loop...")
        qat_model.train() # Set model to training mode for QAT
        optimizer = torch.optim.SGD(qat_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        train_loader = self._get_dummy_dataloader(batch_size=2, num_batches=3, in_channels=1, img_size=28)

        num_epochs = 2
        for epoch in range(num_epochs):
            for i, (inputs,) in enumerate(train_loader):
                inputs = inputs.to(device)
                # Create dummy labels
                dummy_labels = torch.randint(0, 10, (inputs.size(0),), device=device)
                
                optimizer.zero_grad()
                outputs = qat_model(inputs)
                loss = criterion(outputs, dummy_labels)
                loss.backward()
                optimizer.step()
                # print(f"    Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")
        print(f"  Simulated QAT training for {num_epochs} epochs: COMPLETED")

        # 4. (Optional but good practice after QAT) Finalize stub parameters and convert to eval
        print("  Finalizing stub parameters and converting to eval mode...")
        # In a real scenario, you might run one last epoch with stats collection or use EMA stats.
        # For this test, we'll just call compute_quant_params on all stubs.
        # Ensure model is in .train() for compute_quant_params to use the latest min/max
        qat_model.train() 
        for name, module in qat_model.named_modules():
            if isinstance(module, FP4QuantStub):
                module.compute_quant_params() # Use final observed stats
        
        eval_qat_model = convert_ptq_model_to_eval(qat_model) # This just calls .eval()
        self.assertFalse(eval_qat_model.training)
        print("  QAT model finalized and converted to eval: PASSED")

        # 5. Perform a forward pass on the QAT model in eval mode
        print("  Performing forward pass on QAT eval model...")
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
        try:
            with torch.no_grad():
                output = eval_qat_model(dummy_input)
            self.assertEqual(output.shape, (1, 10))
            print("  Forward pass on QAT eval model: PASSED")
        except Exception as e:
            self.fail(f"Forward pass on QAT eval model failed: {e}")

        print("QAT workflow test: COMPLETED")

if __name__ == '__main__':
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 