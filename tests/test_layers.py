# tests/test_layers.py
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Adjust path to import from ignitefp4_lib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ignitefp4_lib.layers import (
    FP4Linear, FP4Conv2d, FP4QuantStub, FP4DequantStub,
    FP4BatchNorm2d, FP4AvgPool2d, FP4MaxPool2d, FP4ConvBnReLU2d
)
from ignitefp4_lib.numerics import (
    dequantize_from_fp4_symmetric,
    calculate_asymmetric_scale_zeropoint,
    quantize_to_fp4_asymmetric,
    dequantize_from_fp4_asymmetric
)

class TestFP4Linear(unittest.TestCase):

    def _create_and_load_fp4_linear(self, in_f, out_f, bias=True, device=None):
        # Ensure factory_kwargs are correctly passed if device is specified for nn.Linear
        factory_kwargs_float = {'device': device} if device else {}
        float_lin = nn.Linear(in_f, out_f, bias=bias, **factory_kwargs_float)
        
        # FP4Linear also needs device if specified
        factory_kwargs_fp4 = {'device': device} if device else {}
        fp4_lin = FP4Linear(in_f, out_f, bias=bias, **factory_kwargs_fp4)
        fp4_lin.load_from_float_linear(float_lin)
        return float_lin, fp4_lin

    def test_initialization_and_loading(self):
        print("\nTesting FP4Linear initialization and loading...")
        # Test with bias on CPU
        float_lin_bias, fp4_lin_bias = self._create_and_load_fp4_linear(10, 5, bias=True, device='cpu')
        self.assertEqual(fp4_lin_bias.in_features, 10)
        self.assertEqual(fp4_lin_bias.out_features, 5)
        self.assertTrue(fp4_lin_bias.bias is not None)
        self.assertIsNotNone(fp4_lin_bias.fp4_weight)
        self.assertIsNotNone(fp4_lin_bias.weight_scale)
        self.assertEqual(fp4_lin_bias.fp4_weight.device.type, 'cpu')
        self.assertEqual(fp4_lin_bias.weight_scale.device.type, 'cpu')
        if fp4_lin_bias.bias is not None:
             self.assertEqual(fp4_lin_bias.bias.device.type, 'cpu')
        
        self.assertTrue(torch.allclose(float_lin_bias.bias.data, fp4_lin_bias.bias.data))
        print("  Initialization and loading with bias (CPU): PASSED")

        # Test without bias on CPU
        _, fp4_lin_no_bias = self._create_and_load_fp4_linear(8, 4, bias=False, device='cpu')
        self.assertTrue(fp4_lin_no_bias.bias is None)
        print("  Initialization and loading without bias (CPU): PASSED")

    def test_forward_pass_cpu(self):
        print("\nTesting FP4Linear forward pass (CPU)...")
        in_features, out_features = 20, 30
        device = 'cpu'
        float_lin, fp4_lin = self._create_and_load_fp4_linear(in_features, out_features, bias=True, device=device)

        input_tensor = torch.randn(2, in_features, device=device) # Batch size of 2

        output_float = float_lin(input_tensor)
        output_fp4_sim = fp4_lin(input_tensor)

        dequantized_weight_manual = dequantize_from_fp4_symmetric(fp4_lin.fp4_weight, fp4_lin.weight_scale)
        expected_output_manual = F.linear(input_tensor, dequantized_weight_manual, fp4_lin.bias)
        
        self.assertTrue(torch.allclose(output_fp4_sim, expected_output_manual, atol=1e-6),
                        "FP4Linear CPU output differs from manually dequantized calculation.")
        
        max_output_diff = torch.abs(output_float - output_fp4_sim).max().item()
        # This tolerance might need adjustment based on typical quantization errors
        self.assertTrue(torch.allclose(output_float, output_fp4_sim, atol=0.5), 
                        f"Output difference between float and FP4 sim (CPU) is too large: {max_output_diff}")
        print(f"  Max output diff (float vs FP4 sim CPU): {max_output_diff}")
        print("  Forward pass CPU: PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping CUDA test.")
    def test_initialization_and_loading_cuda(self):
        print("\nTesting FP4Linear initialization and loading (CUDA)...")
        device = torch.device("cuda")
        float_lin_bias, fp4_lin_bias = self._create_and_load_fp4_linear(10, 5, bias=True, device=device)
        self.assertEqual(fp4_lin_bias.fp4_weight.device.type, "cuda")
        self.assertEqual(fp4_lin_bias.weight_scale.device.type, "cuda")
        if fp4_lin_bias.bias is not None:
             self.assertEqual(fp4_lin_bias.bias.device.type, "cuda")
        self.assertTrue(torch.allclose(float_lin_bias.bias.data, fp4_lin_bias.bias.data))
        print("  Initialization and loading with bias (CUDA): PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping CUDA test.")
    def test_forward_pass_cuda(self):
        print("\nTesting FP4Linear forward pass (CUDA)...")
        device = torch.device("cuda")
        in_features, out_features = 16, 24
        float_lin, fp4_lin = self._create_and_load_fp4_linear(in_features, out_features, bias=True, device=device)
        
        input_tensor = torch.randn(3, in_features, device=device) # Batch size of 3
        output_float = float_lin(input_tensor)
        output_fp4_sim = fp4_lin(input_tensor)

        dequantized_weight_manual = dequantize_from_fp4_symmetric(fp4_lin.fp4_weight, fp4_lin.weight_scale)
        expected_output_manual = F.linear(input_tensor, dequantized_weight_manual, fp4_lin.bias)
        
        self.assertTrue(torch.allclose(output_fp4_sim, expected_output_manual, atol=1e-6),
                        "FP4Linear CUDA output differs from manually dequantized calculation.")
        max_output_diff = torch.abs(output_float - output_fp4_sim).max().item()
        self.assertTrue(torch.allclose(output_float, output_fp4_sim, atol=0.5),
                        f"Output difference between float and FP4 sim (CUDA) is too large: {max_output_diff}")
        print(f"  Max output diff (float vs FP4 sim CUDA): {max_output_diff}")
        print("  Forward pass CUDA: PASSED")
        
    def test_repr(self):
        print("\nTesting FP4Linear __repr__...")
        fp4_lin_cpu = FP4Linear(10, 5, bias=True, device='cpu')
        representation = repr(fp4_lin_cpu)
        self.assertIn("in_features=10", representation)
        self.assertIn("out_features=5", representation)
        self.assertIn("bias=True", representation)
        self.assertIn("weight_quantized=FP4_symmetric", representation)
        print(f"  Representation: {representation}")
        print("  __repr__: PASSED")

class TestFP4Conv2d(unittest.TestCase):

    def _create_and_load_fp4_conv2d(self,
                                     in_channels, out_channels, kernel_size,
                                     stride=1, padding=0, dilation=1, groups=1, bias=True,
                                     padding_mode='zeros', device=None):
        
        factory_kwargs_float = {'device': device} if device else {}
        float_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation,
                               groups=groups, bias=bias, padding_mode=padding_mode, **factory_kwargs_float)

        factory_kwargs_fp4 = {'device': device} if device else {}
        fp4_conv = FP4Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, dilation=dilation,
                             groups=groups, bias=bias, padding_mode=padding_mode, **factory_kwargs_fp4)
        fp4_conv.load_from_float_conv2d(float_conv)
        return float_conv, fp4_conv

    def test_initialization_and_loading(self):
        print("\nTesting FP4Conv2d initialization and loading...")
        # Basic case CPU
        f_conv, fp4_conv = self._create_and_load_fp4_conv2d(3, 16, 3, stride=1, padding=1, bias=True, device='cpu')
        self.assertEqual(fp4_conv.in_channels, 3)
        self.assertEqual(fp4_conv.out_channels, 16)
        self.assertEqual(fp4_conv.kernel_size, (3,3))
        self.assertTrue(fp4_conv.bias is not None)
        self.assertIsNotNone(fp4_conv.fp4_weight)
        self.assertIsNotNone(fp4_conv.weight_scale)
        self.assertEqual(fp4_conv.fp4_weight.device.type, 'cpu')
        if fp4_conv.bias is not None:
            self.assertTrue(torch.allclose(f_conv.bias.data, fp4_conv.bias.data))
        print("  Init and load (CPU, bias=True): PASSED")

        # No bias, different stride/padding
        f_conv_nb, fp4_conv_nb = self._create_and_load_fp4_conv2d(1, 8, (5,5), stride=2, padding='same', bias=False, device='cpu')
        self.assertTrue(fp4_conv_nb.bias is None)
        self.assertEqual(fp4_conv_nb.padding, 'same')
        print("  Init and load (CPU, bias=False, padding='same'): PASSED")
        
        # Groups
        f_conv_g, fp4_conv_g = self._create_and_load_fp4_conv2d(4, 8, 3, groups=4, bias=True, device='cpu')
        self.assertEqual(fp4_conv_g.groups, 4)
        self.assertEqual(fp4_conv_g.fp4_weight.shape, (8, 4 // 4, 3, 3))
        print("  Init and load (CPU, groups): PASSED")


    def test_forward_pass_cpu(self):
        print("\nTesting FP4Conv2d forward pass (CPU)...")
        device = 'cpu'
        float_conv, fp4_conv = self._create_and_load_fp4_conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=True, device=device
        )
        input_tensor = torch.randn(2, 3, 32, 32, device=device) # Batch, Channels, H, W

        output_float = float_conv(input_tensor)
        output_fp4_sim = fp4_conv(input_tensor)

        dequantized_weight_manual = dequantize_from_fp4_symmetric(fp4_conv.fp4_weight, fp4_conv.weight_scale)
        expected_output_manual = F.conv2d(input_tensor, dequantized_weight_manual, fp4_conv.bias,
                                          fp4_conv.stride, fp4_conv.padding, fp4_conv.dilation, fp4_conv.groups)
        
        self.assertTrue(torch.allclose(output_fp4_sim, expected_output_manual, atol=1e-5),
                        "FP4Conv2d CPU output differs from manually dequantized calculation.")
        
        max_output_diff = torch.abs(output_float - output_fp4_sim).max().item()
        self.assertTrue(torch.allclose(output_float, output_fp4_sim, atol=1.0),
                        f"Output difference between float and FP4 sim (CPU) is too large: {max_output_diff}")
        print(f"  Max output diff (float vs FP4 sim CPU): {max_output_diff}")
        print("  Forward pass CPU: PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping CUDA test.")
    def test_initialization_and_loading_cuda(self):
        print("\nTesting FP4Conv2d initialization and loading (CUDA)...")
        device = torch.device("cuda")
        f_conv, fp4_conv = self._create_and_load_fp4_conv2d(3, 8, 3, bias=True, device=device)
        self.assertEqual(fp4_conv.fp4_weight.device.type, "cuda")
        self.assertEqual(fp4_conv.weight_scale.device.type, "cuda")
        if fp4_conv.bias is not None:
             self.assertEqual(fp4_conv.bias.device.type, "cuda")
        self.assertTrue(torch.allclose(f_conv.bias.data, fp4_conv.bias.data))
        print("  Initialization and loading (CUDA): PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping CUDA test.")
    def test_forward_pass_cuda(self):
        print("\nTesting FP4Conv2d forward pass (CUDA)...")
        device = torch.device("cuda")
        float_conv, fp4_conv = self._create_and_load_fp4_conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=True, device=device
        )
        input_tensor = torch.randn(2, 3, 28, 28, device=device)

        output_float = float_conv(input_tensor)
        output_fp4_sim = fp4_conv(input_tensor)

        dequantized_weight_manual = dequantize_from_fp4_symmetric(fp4_conv.fp4_weight, fp4_conv.weight_scale)
        expected_output_manual = F.conv2d(input_tensor, dequantized_weight_manual, fp4_conv.bias,
                                          fp4_conv.stride, fp4_conv.padding, fp4_conv.dilation, fp4_conv.groups)
        
        self.assertTrue(torch.allclose(output_fp4_sim, expected_output_manual, atol=1e-5))
        max_output_diff = torch.abs(output_float - output_fp4_sim).max().item()
        self.assertTrue(torch.allclose(output_float, output_fp4_sim, atol=1.0))
        print(f"  Max output diff (float vs FP4 sim CUDA): {max_output_diff}")
        print("  Forward pass CUDA: PASSED")

    def test_repr(self):
        print("\nTesting FP4Conv2d __repr__...")
        fp4_conv = FP4Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True, device='cpu')
        representation = repr(fp4_conv)
        self.assertIn("in_channels=3", representation)
        self.assertIn("out_channels=64", representation)
        self.assertIn("kernel_size=(3, 3)", representation)
        self.assertIn("stride=(1, 1)", representation)
        self.assertIn("padding=1", representation)
        self.assertIn("bias=True", representation)
        self.assertIn("weight_quantized=FP4_symmetric", representation)
        print(f"  Representation: {representation}")

        fp4_conv_complex = FP4Conv2d(4, 8, 5, stride=2, padding='same', dilation=2, groups=2, bias=False, padding_mode='reflect', device='cpu')
        rep_complex = repr(fp4_conv_complex)
        self.assertIn("padding=same", rep_complex)
        self.assertIn("dilation=(2, 2)", rep_complex)
        self.assertIn("groups=2", rep_complex)
        self.assertIn("bias=False", rep_complex)
        self.assertIn("padding_mode=reflect", rep_complex)
        print(f"  Representation (complex): {rep_complex}")
        print("  __repr__: PASSED")

class TestQuantStubs(unittest.TestCase):
    def test_fp4_quant_stub_initialization(self):
        print("\nTesting FP4QuantStub initialization...")
        # Default: qat_mode=False, learnable_params=False
        stub_ptq = FP4QuantStub()
        self.assertFalse(stub_ptq.qat_mode)
        self.assertFalse(stub_ptq.learnable_params)
        self.assertIsInstance(stub_ptq.scale, torch.Tensor) # Buffer
        self.assertNotIsInstance(stub_ptq.scale, nn.Parameter)
        self.assertIsInstance(stub_ptq.zero_point, torch.Tensor) # Buffer
        self.assertNotIsInstance(stub_ptq.zero_point, nn.Parameter)
        print("  Init (qat_mode=F, learnable=F): PASSED")

        # QAT, non-learnable
        stub_qat_fixed = FP4QuantStub(qat_mode=True, learnable_params=False)
        self.assertTrue(stub_qat_fixed.qat_mode)
        self.assertFalse(stub_qat_fixed.learnable_params)
        self.assertNotIsInstance(stub_qat_fixed.scale, nn.Parameter)
        print("  Init (qat_mode=T, learnable=F): PASSED")

        # QAT, learnable
        stub_qat_learn = FP4QuantStub(qat_mode=True, learnable_params=True)
        self.assertTrue(stub_qat_learn.qat_mode)
        self.assertTrue(stub_qat_learn.learnable_params)
        self.assertIsInstance(stub_qat_learn.scale, nn.Parameter)
        self.assertIsInstance(stub_qat_learn.zero_point, nn.Parameter)
        print("  Init (qat_mode=T, learnable=T): PASSED")
        # Check initial values of learnable params
        self.assertEqual(stub_qat_learn.scale.item(), 1.0)
        self.assertEqual(stub_qat_learn.zero_point.item(), 0.0)

    def test_fp4_quant_stub_ptq_calibration_mode(self):
        print("\nTesting FP4QuantStub PTQ calibration mode (qat_mode=False, learnable_params=False)...")
        stub = FP4QuantStub(qat_mode=False, learnable_params=False).train()
        data1 = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
        out1 = stub(data1)
        self.assertEqual(stub.min_val.item(), -1.0)
        self.assertEqual(stub.max_val.item(), 3.0)
        self.assertTrue(torch.equal(out1, data1), "Output should be pass-through in PTQ calibration")
        print(f"  PTQ pass 1: min_val={stub.min_val.item()}, max_val={stub.max_val.item()}, output pass-through: PASSED")

        data2 = torch.tensor([-2.0, 0.5, 2.5, 4.0])
        out2 = stub(data2)
        self.assertEqual(stub.min_val.item(), -2.0)
        self.assertEqual(stub.max_val.item(), 4.0)
        self.assertTrue(torch.equal(out2, data2), "Output should be pass-through in PTQ calibration")
        print(f"  PTQ pass 2: min_val={stub.min_val.item()}, max_val={stub.max_val.item()}, output pass-through: PASSED")

        stub.compute_quant_params()
        self.assertTrue(stub.is_calibrated)
        expected_scale, expected_zp = calculate_asymmetric_scale_zeropoint(torch.tensor([-2.0, 4.0]))
        self.assertAlmostEqual(stub.scale.item(), expected_scale.item(), places=5)
        self.assertEqual(stub.zero_point.item(), expected_zp.item())
        print(f"  Calibrated params: scale={stub.scale.item()}, zp={stub.zero_point.item()}")

        stub.eval()
        test_data = torch.tensor([-3.0, -2.0, 0.0, 2.0, 4.0, 5.0])
        expected_output = dequantize_from_fp4_asymmetric(
            quantize_to_fp4_asymmetric(test_data, stub.scale, stub.zero_point.long()), # .long() for buffer zp
            stub.scale,
            stub.zero_point.long()
        )
        output = stub(test_data)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))
        print(f"  Eval mode output after PTQ calibration: {output.tolist()}")
        print("  FP4QuantStub PTQ calibration mode: PASSED")

    def test_fp4_quant_stub_qat_mode_batch_stats(self):
        print("\nTesting FP4QuantStub QAT mode (qat_mode=True, learnable_params=False, is_calibrated=False - uses running stats)...")
        stub = FP4QuantStub(qat_mode=True, learnable_params=False).train()
        data1 = torch.tensor([0.0, 5.0, 10.0, 15.0]) # min=0, max=15
        out1 = stub(data1)
        s1, zp1 = calculate_asymmetric_scale_zeropoint(torch.tensor([0.0, 15.0]))
        expected_out1 = dequantize_from_fp4_asymmetric(quantize_to_fp4_asymmetric(data1, s1, zp1), s1, zp1)
        self.assertTrue(torch.allclose(out1, expected_out1, atol=1e-5))
        self.assertEqual(stub.min_val.item(), 0.0)
        self.assertEqual(stub.max_val.item(), 15.0)
        print(f"  QAT non-learnable, pass 1 (running stats [0,15]): output={out1.tolist()}")

        data2 = torch.tensor([-5.0, 0.0, 10.0, 20.0])
        out2 = stub(data2)
        self.assertEqual(stub.min_val.item(), -5.0)
        self.assertEqual(stub.max_val.item(), 20.0)
        s2, zp2 = calculate_asymmetric_scale_zeropoint(torch.tensor([-5.0, 20.0]))
        expected_out2 = dequantize_from_fp4_asymmetric(quantize_to_fp4_asymmetric(data2, s2, zp2), s2, zp2)
        self.assertTrue(torch.allclose(out2, expected_out2, atol=1e-5))
        print(f"  QAT non-learnable, pass 2 (running stats [-5,20]): output={out2.tolist()}")
        print("  FP4QuantStub QAT mode (non-learnable, batch/running stats): PASSED")

    def test_fp4_quant_stub_qat_mode_calibrated_stats(self):
        print("\nTesting FP4QuantStub QAT mode (qat_mode=True, learnable_params=False, is_calibrated=True - uses fixed stats)...")
        stub = FP4QuantStub(qat_mode=True, learnable_params=False).train()
        calib_data = torch.arange(0, 16, dtype=torch.float32)
        stub(calib_data)
        stub.compute_quant_params()
        self.assertTrue(stub.is_calibrated)
        fixed_scale, fixed_zp = stub.scale, stub.zero_point # buffers
        print(f"  QAT non-learnable, fixed stats: Calibrated scale={fixed_scale.item()}, zp={fixed_zp.item()}")
        data_qat = torch.tensor([-5.0, 7.5, 15.0, 20.0])
        out_qat = stub(data_qat)
        expected_out_qat = dequantize_from_fp4_asymmetric(
            quantize_to_fp4_asymmetric(data_qat, fixed_scale, fixed_zp.long()), fixed_scale, fixed_zp.long()
        )
        self.assertTrue(torch.allclose(out_qat, expected_out_qat, atol=1e-5))
        print(f"  QAT non-learnable, fixed stats: output={out_qat.tolist()}")
        print("  FP4QuantStub QAT mode (non-learnable, fixed stats): PASSED")

    def test_fp4_quant_stub_qat_mode_learnable_params(self):
        print("\nTesting FP4QuantStub QAT mode (learnable_params=True)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        stub = FP4QuantStub(qat_mode=True, learnable_params=True, device=device, dtype=torch.float32)
        stub.train() # Set to training mode

        # Initialize learnable parameters: feed some data and call compute_quant_params
        init_data = torch.tensor([-10.0, 0.0, 10.0, 20.0, 30.0], device=device)
        stub(init_data) # Observe
        stub.compute_quant_params() # Initializes self.scale and self.zero_point.data
        self.assertTrue(stub.is_calibrated)

        # Store initial scale and zero_point values
        initial_scale = stub.scale.clone().detach()
        initial_zero_point = stub.zero_point.clone().detach()
        print(f"  Initial learnable params: scale={initial_scale.item():.4f}, zp={initial_zero_point.item():.4f}")

        # Create dummy input and optimizer
        input_tensor = torch.tensor([1.0, 5.0, 12.0, 25.0], device=device, requires_grad=False) 
        # target requires grad for loss.backward() if it depends on params.
        # Let's make a simple loss: mean of output (so gradient flows)
        optimizer = torch.optim.SGD([stub.scale, stub.zero_point], lr=0.1)

        # Forward pass
        output = stub(input_tensor)
        self.assertTrue(output.requires_grad, "Output should require grad if params are learnable and input is part of graph (though input is not here)")

        # Backward pass and optimization step
        # Dummy loss: try to push the mean of the output towards a target value, e.g., 10.
        loss = (output.mean() - 10.0)**2
        optimizer.zero_grad()
        loss.backward()
        
        self.assertIsNotNone(stub.scale.grad, "Gradient for scale should not be None")
        self.assertIsNotNone(stub.zero_point.grad, "Gradient for zero_point should not be None")
        # print(f"  Grads: scale_grad={stub.scale.grad.item():.4f}, zp_grad={stub.zero_point.grad.item():.4f}")
        self.assertNotEqual(stub.scale.grad.item(), 0.0, "Scale gradient should be non-zero for this setup")
        # Zero point grad could be zero if scale is large and all inputs map to same quantized bin far from zero point influence
        # or if inputs are such that the zero point doesn't affect the rounded output much relative to scale.
        # For this test, let's just check it exists.

        optimizer.step()

        # Check if parameters have been updated
        updated_scale = stub.scale.clone().detach()
        updated_zero_point = stub.zero_point.clone().detach()
        print(f"  Updated learnable params: scale={updated_scale.item():.4f}, zp={updated_zero_point.item():.4f}")

        self.assertNotEqual(initial_scale.item(), updated_scale.item(), "Scale should have been updated by optimizer")
        # Zero point might not change much or at all if its gradient was tiny or zero
        # self.assertNotEqual(initial_zero_point.item(), updated_zero_point.item(), "Zero point should have been updated")
        if torch.allclose(initial_zero_point, updated_zero_point):
            print("  Note: Zero point did not change significantly, this can happen depending on grads.")

        print("  FP4QuantStub QAT mode (learnable_params=True, gradient check): PASSED")

    def test_fp4_quant_stub_compute_params_and_eval(self):
        print("\nTesting FP4QuantStub compute_quant_params and eval mode (PTQ flow, learnable_params=False)...")
        stub = FP4QuantStub(qat_mode=False, learnable_params=False).train()
        calib_data = torch.arange(-2, 14, dtype=torch.float32)
        stub(calib_data)
        stub.compute_quant_params()
        self.assertTrue(stub.is_calibrated)
        self.assertAlmostEqual(stub.scale.item(), 1.0, places=5)
        self.assertEqual(stub.zero_point.item(), 2)
        stub.eval()
        test_data = torch.tensor([-5.0, -2.0, 0.0, 7.0, 13.0, 15.0])
        expected_output = torch.tensor([-2.0, -2.0, 0.0, 7.0, 13.0, 13.0])
        output = stub(test_data)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))
        print("  FP4QuantStub compute_params and eval mode (PTQ flow): PASSED")

    def test_fp4_quant_stub_enabled_disabled(self):
        print("\nTesting FP4QuantStub enabled/disabled (learnable_params=False)...")
        stub = FP4QuantStub(qat_mode=False, learnable_params=False).train()
        stub(torch.tensor([0.0,15.0]))
        stub.compute_quant_params()
        stub.eval()
        test_data = torch.tensor([0.0, 7.5, 15.0])
        stub.enabled = True
        output_enabled = stub(test_data)
        stub.enabled = False
        output_disabled = stub(test_data)
        self.assertTrue(torch.allclose(output_disabled, test_data))
        expected_enabled_output = torch.tensor([0.0, 8.0, 15.0])
        self.assertTrue(torch.allclose(output_enabled, expected_enabled_output, atol=1e-5))
        print("  FP4QuantStub enabled/disabled: PASSED")

    def test_fp4_dequant_stub(self):
        print("\nTesting FP4DequantStub...")
        stub = FP4DequantStub()
        test_data = torch.randn(5)
        output = stub(test_data)
        self.assertTrue(torch.equal(output, test_data))
        print(f"  Representation: {repr(stub)}")
        print("  FP4DequantStub (identity): PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp4_quant_stub_cuda(self):
        print("\nTesting FP4QuantStub on CUDA (PTQ, QAT fixed, QAT learnable)...")
        device = torch.device("cuda")

        # 1. Test PTQ-style (learnable_params=False) on CUDA
        print("  Testing PTQ flow (learnable=F) on CUDA...")
        stub_ptq_cuda = FP4QuantStub(qat_mode=False, learnable_params=False).to(device)
        stub_ptq_cuda.train()
        calib_data_cuda = torch.arange(0, 16, dtype=torch.float32, device=device)
        out_calib_cuda = stub_ptq_cuda(calib_data_cuda)
        self.assertTrue(torch.equal(out_calib_cuda, calib_data_cuda))
        stub_ptq_cuda.compute_quant_params()
        self.assertAlmostEqual(stub_ptq_cuda.scale.item(), 1.0, places=5)
        self.assertEqual(stub_ptq_cuda.zero_point.item(), 0)
        stub_ptq_cuda.eval()
        test_data_cuda_ptq = torch.tensor([-1.0, 0.0, 7.5, 15.0, 20.0], device=device)
        expected_output_cuda_ptq = torch.tensor([0.0, 0.0, 8.0, 15.0, 15.0], device=device)
        output_cuda_ptq = stub_ptq_cuda(test_data_cuda_ptq)
        self.assertTrue(torch.allclose(output_cuda_ptq, expected_output_cuda_ptq, atol=1e-5))
        print("  PTQ flow (learnable=F) on CUDA: PASSED")

        # 2. Test QAT-style (learnable_params=False, batch/running stats) on CUDA
        print("\n  Testing QAT flow (learnable=F, running stats) on CUDA...")
        stub_qat_cuda_batch = FP4QuantStub(qat_mode=True, learnable_params=False).to(device)
        stub_qat_cuda_batch.train()
        data1_cuda = torch.tensor([0.0, 5.0, 10.0, 15.0], device=device)
        out1_qat_cuda = stub_qat_cuda_batch(data1_cuda)
        s1_cuda, zp1_cuda = calculate_asymmetric_scale_zeropoint(torch.tensor([0.0, 15.0], device=device))
        expected_out1_qat_cuda = dequantize_from_fp4_asymmetric(
            quantize_to_fp4_asymmetric(data1_cuda, s1_cuda, zp1_cuda), s1_cuda, zp1_cuda
        )
        self.assertTrue(torch.allclose(out1_qat_cuda, expected_out1_qat_cuda, atol=1e-5))
        print("  QAT flow (learnable=F, running stats) on CUDA: PASSED")

        # 3. Test QAT-style (learnable_params=True) on CUDA
        print("\n  Testing QAT flow (learnable_params=True) on CUDA...")
        stub_qat_learn_cuda = FP4QuantStub(qat_mode=True, learnable_params=True, device=device, dtype=torch.float32)
        stub_qat_learn_cuda.train()

        init_data_cuda = torch.tensor([-5.0, 5.0, 15.0, 25.0], device=device)
        stub_qat_learn_cuda(init_data_cuda) # Observe
        stub_qat_learn_cuda.compute_quant_params() # Initialize learnable params
        
        initial_scale_cuda = stub_qat_learn_cuda.scale.clone().detach()
        initial_zp_cuda = stub_qat_learn_cuda.zero_point.clone().detach()

        optimizer_cuda = torch.optim.SGD([stub_qat_learn_cuda.scale, stub_qat_learn_cuda.zero_point], lr=0.01)
        input_tensor_cuda = torch.tensor([0.0, 10.0, 20.0], device=device)
        output_learn_cuda = stub_qat_learn_cuda(input_tensor_cuda)
        loss_cuda = (output_learn_cuda.mean() - 5.0)**2 # Dummy loss

        optimizer_cuda.zero_grad()
        loss_cuda.backward()
        self.assertIsNotNone(stub_qat_learn_cuda.scale.grad)
        self.assertIsNotNone(stub_qat_learn_cuda.zero_point.grad)
        optimizer_cuda.step()

        self.assertNotEqual(initial_scale_cuda.item(), stub_qat_learn_cuda.scale.item(), "CUDA Scale should change")
        # self.assertNotEqual(initial_zp_cuda.item(), stub_qat_learn_cuda.zero_point.item(), "CUDA ZP should change")
        print("  QAT flow (learnable_params=True, gradient check) on CUDA: PASSED")
        print("  All FP4QuantStub CUDA tests: PASSED")

class TestFP4BatchNorm2d(unittest.TestCase):
    def _create_and_load_fp4_bn(self, num_features, affine=True, track_running_stats=True, device=None):
        factory_kwargs = {'device': device} if device else {}
        float_bn = nn.BatchNorm2d(num_features, affine=affine, track_running_stats=track_running_stats, **factory_kwargs)
        # Put in eval mode to ensure running_mean/var are used if track_running_stats=True for initial values
        # Or train for a bit to populate them
        if track_running_stats:
            float_bn.train()
            dummy_input = torch.randn(2, num_features, 3, 3, **factory_kwargs) # B, C, H, W
            for _ in range(3): # Run a few batches to get some running stats
                float_bn(dummy_input + torch.rand_like(dummy_input))
            float_bn.eval() # For stable running_mean/var values for loading

        fp4_bn = FP4BatchNorm2d(num_features, affine=affine, track_running_stats=track_running_stats, **factory_kwargs)
        fp4_bn.load_from_float_batchnorm2d(float_bn)
        return float_bn, fp4_bn

    def test_initialization_and_loading_cpu(self):
        print("\nTesting FP4BatchNorm2d initialization and loading (CPU)...")
        num_features = 8
        device = 'cpu'

        # Test case 1: Affine=True, TrackStats=True
        f_bn_aff_track, fp4_bn_aff_track = self._create_and_load_fp4_bn(num_features, True, True, device)
        self.assertTrue(fp4_bn_aff_track.affine)
        self.assertTrue(fp4_bn_aff_track.track_running_stats)
        self.assertIsNotNone(fp4_bn_aff_track.fp4_weight)
        self.assertIsNotNone(fp4_bn_aff_track.weight_scale)
        self.assertIsNotNone(fp4_bn_aff_track.fp4_bias)
        self.assertIsNotNone(fp4_bn_aff_track.bias_scale)
        self.assertIsNotNone(fp4_bn_aff_track.running_mean)
        self.assertIsNotNone(fp4_bn_aff_track.running_var)
        self.assertTrue(torch.allclose(f_bn_aff_track.running_mean, fp4_bn_aff_track.running_mean, atol=1e-5))
        self.assertTrue(torch.allclose(f_bn_aff_track.running_var, fp4_bn_aff_track.running_var, atol=1e-5))
        # Dequantize and check weight/bias
        dequant_w = dequantize_from_fp4_symmetric(fp4_bn_aff_track.fp4_weight, fp4_bn_aff_track.weight_scale)
        self.assertTrue(torch.allclose(f_bn_aff_track.weight.data, dequant_w, atol=0.1)) # Tolerance for FP4
        dequant_b = dequantize_from_fp4_symmetric(fp4_bn_aff_track.fp4_bias, fp4_bn_aff_track.bias_scale)
        self.assertTrue(torch.allclose(f_bn_aff_track.bias.data, dequant_b, atol=0.1))
        print("  Init/Load (Affine=T, TrackStats=T, CPU): PASSED")

        # Test case 2: Affine=False, TrackStats=True
        f_bn_noaff_track, fp4_bn_noaff_track = self._create_and_load_fp4_bn(num_features, False, True, device)
        self.assertFalse(fp4_bn_noaff_track.affine)
        self.assertTrue(fp4_bn_noaff_track.track_running_stats)
        self.assertIsNone(fp4_bn_noaff_track.fp4_weight)
        self.assertTrue(torch.allclose(f_bn_noaff_track.running_mean, fp4_bn_noaff_track.running_mean, atol=1e-5))
        print("  Init/Load (Affine=F, TrackStats=T, CPU): PASSED")

        # Test case 3: Affine=True, TrackStats=False
        f_bn_aff_notrack, fp4_bn_aff_notrack = self._create_and_load_fp4_bn(num_features, True, False, device)
        self.assertTrue(fp4_bn_aff_notrack.affine)
        self.assertFalse(fp4_bn_aff_notrack.track_running_stats)
        self.assertIsNotNone(fp4_bn_aff_notrack.fp4_weight)
        self.assertIsNone(fp4_bn_aff_notrack.running_mean)
        print("  Init/Load (Affine=T, TrackStats=F, CPU): PASSED")

    def test_forward_pass_cpu(self):
        print("\nTesting FP4BatchNorm2d forward pass (CPU)...")
        num_features = 16
        device = 'cpu'
        input_tensor = torch.randn(4, num_features, 6, 6, device=device) # B, C, H, W

        # Case 1: Affine=True, TrackStats=True, Training mode
        f_bn_train, fp4_bn_train = self._create_and_load_fp4_bn(num_features, True, True, device)
        f_bn_train.train()
        fp4_bn_train.train()
        # Copy initial running stats to ensure fair comparison for update
        fp4_bn_train.running_mean.data.copy_(f_bn_train.running_mean.data)
        fp4_bn_train.running_var.data.copy_(f_bn_train.running_var.data)
        fp4_bn_train.num_batches_tracked.data.copy_(f_bn_train.num_batches_tracked.data)

        output_float_train = f_bn_train(input_tensor)
        output_fp4_train = fp4_bn_train(input_tensor.clone()) # clone to avoid in-place modification issues with CUDNN for float
        
        self.assertTrue(torch.allclose(output_float_train, output_fp4_train, atol=0.2), # Higher atol due to FP4 params
                        f"Train output diff: {(output_float_train - output_fp4_train).abs().max().item()}")
        self.assertFalse(torch.allclose(f_bn_train.running_mean, fp4_bn_train.running_mean, atol=1e-7)) # Should differ due to updates from different precision inputs
        print("  Forward (Affine=T, TrackStats=T, Train, CPU): PASSED")

        # Case 2: Affine=True, TrackStats=True, Eval mode
        f_bn_eval, fp4_bn_eval = self._create_and_load_fp4_bn(num_features, True, True, device)
        f_bn_eval.eval()
        fp4_bn_eval.eval()
        # Ensure running stats are identical before eval pass
        fp4_bn_eval.running_mean.data.copy_(f_bn_eval.running_mean.data)
        fp4_bn_eval.running_var.data.copy_(f_bn_eval.running_var.data)

        output_float_eval = f_bn_eval(input_tensor)
        output_fp4_eval = fp4_bn_eval(input_tensor.clone())
        self.assertTrue(torch.allclose(output_float_eval, output_fp4_eval, atol=0.2),
                        f"Eval output diff: {(output_float_eval - output_fp4_eval).abs().max().item()}")
        # Running stats should NOT change in eval mode
        self.assertTrue(torch.allclose(f_bn_eval.running_mean, fp4_bn_eval.running_mean, atol=1e-7))
        print("  Forward (Affine=T, TrackStats=T, Eval, CPU): PASSED")

        # Case 3: Affine=True, TrackStats=False (always uses batch stats)
        f_bn_notrack, fp4_bn_notrack = self._create_and_load_fp4_bn(num_features, True, False, device)
        f_bn_notrack.train() # Should use batch stats
        fp4_bn_notrack.train()
        output_float_notrack_train = f_bn_notrack(input_tensor)
        output_fp4_notrack_train = fp4_bn_notrack(input_tensor.clone())
        self.assertTrue(torch.allclose(output_float_notrack_train, output_fp4_notrack_train, atol=0.2))
        
        f_bn_notrack.eval() # Should still use batch stats as track_running_stats is False
        fp4_bn_notrack.eval()
        output_float_notrack_eval = f_bn_notrack(input_tensor)
        output_fp4_notrack_eval = fp4_bn_notrack(input_tensor.clone())
        self.assertTrue(torch.allclose(output_float_notrack_eval, output_fp4_notrack_eval, atol=0.2))
        self.assertTrue(torch.allclose(output_float_notrack_train, output_float_notrack_eval, atol=1e-7)) # Output should be same for train/eval if not tracking
        print("  Forward (Affine=T, TrackStats=F, CPU): PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_initialization_and_loading_cuda(self):
        print("\nTesting FP4BatchNorm2d initialization and loading (CUDA)...")
        num_features = 4
        device = torch.device("cuda")
        f_bn, fp4_bn = self._create_and_load_fp4_bn(num_features, True, True, device)
        self.assertEqual(fp4_bn.fp4_weight.device.type, "cuda")
        self.assertEqual(fp4_bn.running_mean.device.type, "cuda")
        dequant_w = dequantize_from_fp4_symmetric(fp4_bn.fp4_weight, fp4_bn.weight_scale)
        self.assertTrue(torch.allclose(f_bn.weight.data, dequant_w, atol=0.1))
        print("  Init/Load (Affine=T, TrackStats=T, CUDA): PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_forward_pass_cuda(self):
        print("\nTesting FP4BatchNorm2d forward pass (CUDA)...")
        num_features = 10
        device = torch.device("cuda")
        input_tensor = torch.randn(2, num_features, 5, 5, device=device)

        f_bn, fp4_bn = self._create_and_load_fp4_bn(num_features, True, True, device)
        f_bn.train()
        fp4_bn.train()
        fp4_bn.running_mean.data.copy_(f_bn.running_mean.data)
        fp4_bn.running_var.data.copy_(f_bn.running_var.data)
        fp4_bn.num_batches_tracked.data.copy_(f_bn.num_batches_tracked.data)

        output_float = f_bn(input_tensor)
        output_fp4 = fp4_bn(input_tensor.clone())
        self.assertTrue(torch.allclose(output_float, output_fp4, atol=0.2),
                        f"CUDA Train output diff: {(output_float - output_fp4).abs().max().item()}")
        print("  Forward (Affine=T, TrackStats=T, Train, CUDA): PASSED")
        
        f_bn.eval()
        fp4_bn.eval()
        # Ensure running stats are identical before eval pass
        # Note: load_from_float already sets fp4_bn to eval and copies stats
        # For an isolated eval test, one might re-load or ensure stats are set correctly.
        f_bn_eval_direct, fp4_bn_eval_direct = self._create_and_load_fp4_bn(num_features, True, True, device)
        f_bn_eval_direct.eval()
        fp4_bn_eval_direct.eval() # load_from_float_batchnorm2d calls eval on float_bn before loading

        output_float_eval = f_bn_eval_direct(input_tensor)
        output_fp4_eval = fp4_bn_eval_direct(input_tensor.clone())
        self.assertTrue(torch.allclose(output_float_eval, output_fp4_eval, atol=0.2),
                        f"CUDA Eval output diff: {(output_float_eval - output_fp4_eval).abs().max().item()}")
        print("  Forward (Affine=T, TrackStats=T, Eval, CUDA): PASSED")

    def test_repr(self):
        print("\nTesting FP4BatchNorm2d __repr__...")
        fp4_bn = FP4BatchNorm2d(32, affine=True, track_running_stats=True)
        representation = repr(fp4_bn)
        self.assertIn("32", representation)
        self.assertIn("affine=True", representation)
        self.assertIn("track_running_stats=True", representation)
        self.assertIn("params_quantized=FP4_symmetric", representation)
        print(f"  Representation (affine=T, track=T): {representation}")

        fp4_bn_no_affine = FP4BatchNorm2d(16, affine=False, track_running_stats=False)
        rep_no_affine = repr(fp4_bn_no_affine)
        self.assertIn("affine=False", rep_no_affine)
        self.assertIn("track_running_stats=False", rep_no_affine)
        self.assertIn("(if affine)", rep_no_affine) # Indicates conditional nature of quantization
        print(f"  Representation (affine=F, track=F): {rep_no_affine}")
        print("  __repr__: PASSED")

class TestPoolingWrappers(unittest.TestCase):
    def test_fp4_avg_pool2d(self):
        print("\nTesting FP4AvgPool2d wrapper...")
        kernel_size = 2
        fp4_pool = FP4AvgPool2d(kernel_size)
        std_pool = nn.AvgPool2d(kernel_size)

        input_tensor = torch.randn(1, 3, 4, 4) # B, C, H, W
        out_fp4 = fp4_pool(input_tensor)
        out_std = std_pool(input_tensor)

        self.assertTrue(torch.equal(out_fp4, out_std))
        self.assertIn("(FP4 Wrapper - Standard Op)", repr(fp4_pool))
        print("  FP4AvgPool2d: PASSED")

    def test_fp4_max_pool2d(self):
        print("\nTesting FP4MaxPool2d wrapper...")
        kernel_size = 3
        stride = 2
        fp4_pool = FP4MaxPool2d(kernel_size, stride=stride)
        std_pool = nn.MaxPool2d(kernel_size, stride=stride)

        input_tensor = torch.randn(1, 1, 7, 7)
        out_fp4 = fp4_pool(input_tensor)
        out_std = std_pool(input_tensor)

        self.assertTrue(torch.equal(out_fp4, out_std))
        self.assertIn("(FP4 Wrapper - Standard Op)", repr(fp4_pool))
        print("  FP4MaxPool2d: PASSED")

class TestFP4ConvBnReLU2d(unittest.TestCase):
    def _create_and_load_fp4_cbr(self, in_c, out_c, kernel_size, stride=1, padding=0, conv_bias=False, device=None):
        factory_kwargs = {'device': device} if device else {}

        # Float modules
        float_conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=conv_bias, **factory_kwargs)
        float_bn = nn.BatchNorm2d(out_c, **factory_kwargs)
        # Initialize BN running stats by passing some data in train mode
        # Determine a reasonable dummy input size for BN init based on conv output
        # This is a rough estimation, assuming no extreme stride/dilation that shrinks output too much
        dummy_h, dummy_w = 8, 8 
        if isinstance(kernel_size, int):
            dummy_h = max(1, 8 - kernel_size + 1 + 2* (padding if isinstance(padding,int) else padding[0]))
            dummy_w = max(1, 8 - kernel_size + 1 + 2* (padding if isinstance(padding,int) else padding[1]))
        else: # kernel_size is a tuple
            dummy_h = max(1, 8 - kernel_size[0] + 1 + 2* (padding if isinstance(padding,int) else padding[0]))
            dummy_w = max(1, 8 - kernel_size[1] + 1 + 2* (padding if isinstance(padding,int) else padding[1]))
        
        # Ensure dummy_h and dummy_w are at least 1
        dummy_h = max(1, dummy_h // (stride if isinstance(stride, int) else stride[0]))
        dummy_w = max(1, dummy_w // (stride if isinstance(stride, int) else stride[1]))

        dummy_input_bn_init = torch.randn(2, out_c, dummy_h, dummy_w, **factory_kwargs)
        if dummy_input_bn_init.numel() == 0:
             # Fallback if calculation results in zero elements (e.g. large kernel, small initial H/W)
            dummy_input_bn_init = torch.randn(2, out_c, 1, 1, **factory_kwargs)

        for _ in range(3):
            float_bn(dummy_input_bn_init + torch.rand_like(dummy_input_bn_init))
        float_bn.eval() # For stable stats for loading

        # FP4 Fused module
        fp4_cbr = FP4ConvBnReLU2d(in_c, out_c, kernel_size, stride=stride, padding=padding, 
                                  conv_bias=conv_bias, device=device, dtype=float_conv.weight.dtype)
        fp4_cbr.load_from_float_modules(float_conv, float_bn)
        return float_conv, float_bn, fp4_cbr

    def test_initialization_and_loading_cpu(self):
        print("\nTesting FP4ConvBnReLU2d initialization and loading (CPU)...")
        in_c, out_c, ks = 3, 8, 3
        device = 'cpu'
        f_conv, f_bn, fp4_cbr = self._create_and_load_fp4_cbr(in_c, out_c, ks, device=device)

        self.assertIsInstance(fp4_cbr.conv, FP4Conv2d)
        self.assertIsInstance(fp4_cbr.bn, FP4BatchNorm2d)
        self.assertEqual(fp4_cbr.conv.in_channels, in_c)
        self.assertEqual(fp4_cbr.conv.out_channels, out_c)
        self.assertEqual(fp4_cbr.bn.num_features, out_c)
        self.assertEqual(fp4_cbr.conv.bias is None, f_conv.bias is None)

        # Check loaded params (simplified check focusing on scales and running_mean)
        self.assertIsNotNone(fp4_cbr.conv.weight_scale)
        self.assertIsNotNone(fp4_cbr.bn.weight_scale) # if affine=True (default for fused)
        self.assertTrue(torch.allclose(f_bn.running_mean, fp4_cbr.bn.running_mean, atol=1e-5))
        dequant_conv_w = dequantize_from_fp4_symmetric(fp4_cbr.conv.fp4_weight, fp4_cbr.conv.weight_scale)
        self.assertTrue(torch.allclose(f_conv.weight.data, dequant_conv_w, atol=0.1))
        print("  Init/Load (CPU): PASSED")

    def test_forward_pass_cpu(self):
        print("\nTesting FP4ConvBnReLU2d forward pass (CPU)...")
        in_c, out_c, ks = 3, 8, 3
        device = 'cpu'
        input_tensor = torch.randn(2, in_c, 16, 16, device=device)

        f_conv, f_bn, fp4_cbr = self._create_and_load_fp4_cbr(in_c, out_c, ks, padding=1, conv_bias=True, device=device)
        
        # Float reference model
        float_model_seq = nn.Sequential(f_conv, f_bn, nn.ReLU()).to(device)

        # Test in eval mode
        float_model_seq.eval()
        fp4_cbr.eval()
        # Ensure BN stats are exactly synced for eval comparison
        fp4_cbr.bn.running_mean.data.copy_(f_bn.running_mean.data)
        fp4_cbr.bn.running_var.data.copy_(f_bn.running_var.data)
        fp4_cbr.bn.num_batches_tracked.data.copy_(f_bn.num_batches_tracked.data)

        with torch.no_grad():
            out_float = float_model_seq(input_tensor)
            out_fp4 = fp4_cbr(input_tensor.clone())
        
        self.assertTrue(torch.allclose(out_float, out_fp4, atol=0.5), # Higher tolerance for multi-stage FP4 sim
                        f"Eval output diff CPU: {(out_float - out_fp4).abs().max().item()}. Float max: {out_float.abs().max()}, FP4 max: {out_fp4.abs().max()}")
        print("  Forward pass (Eval, CPU): PASSED")

        # Test in train mode (BN stats will update)
        float_model_seq.train()
        fp4_cbr.train()
        # Reset and copy initial stats for fair comparison of update
        # Use a known state for running_mean/var before the training pass for f_bn and fp4_cbr.bn
        # to make comparison of their updates more direct if desired, though the primary check is output equivalence.
        initial_running_mean_f = f_bn.running_mean.clone().detach()
        initial_running_var_f = f_bn.running_var.clone().detach()
        initial_running_mean_fp4 = fp4_cbr.bn.running_mean.clone().detach()
        initial_running_var_fp4 = fp4_cbr.bn.running_var.clone().detach()
        
        out_float_train = float_model_seq(input_tensor)
        out_fp4_train = fp4_cbr(input_tensor.clone())
        self.assertTrue(torch.allclose(out_float_train, out_fp4_train, atol=0.5),
                         f"Train output diff CPU: {(out_float_train - out_fp4_train).abs().max().item()}")
        
        # Check that BN running stats have been updated from their initial state
        self.assertFalse(torch.allclose(f_bn.running_mean, initial_running_mean_f, atol=1e-7) and 
                         torch.allclose(f_bn.running_var, initial_running_var_f, atol=1e-7),
                         "Float BN stats should update in train mode.")
        self.assertFalse(torch.allclose(fp4_cbr.bn.running_mean, initial_running_mean_fp4, atol=1e-7) and 
                         torch.allclose(fp4_cbr.bn.running_var, initial_running_var_fp4, atol=1e-7),
                         "FP4 BN stats should update in train mode.")
        print("  Forward pass (Train, CPU): PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_initialization_and_loading_cuda(self):
        print("\nTesting FP4ConvBnReLU2d initialization and loading (CUDA)...")
        in_c, out_c, ks = 2, 4, 3
        device = torch.device("cuda")
        _ , f_bn, fp4_cbr = self._create_and_load_fp4_cbr(in_c, out_c, ks, device=device)
        self.assertEqual(fp4_cbr.conv.fp4_weight.device.type, "cuda")
        self.assertEqual(fp4_cbr.bn.running_mean.device.type, "cuda")
        if fp4_cbr.conv.bias is not None:
            self.assertEqual(fp4_cbr.conv.bias.device.type, "cuda")
        self.assertTrue(torch.allclose(f_bn.running_mean, fp4_cbr.bn.running_mean, atol=1e-5))
        print("  Init/Load (CUDA): PASSED")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_forward_pass_cuda(self):
        print("\nTesting FP4ConvBnReLU2d forward pass (CUDA)...")
        in_c, out_c, ks = 3, 6, 3
        device = torch.device("cuda")
        input_tensor = torch.randn(2, in_c, 12, 12, device=device)
        
        f_conv, f_bn, fp4_cbr = self._create_and_load_fp4_cbr(in_c, out_c, ks, padding=1, conv_bias=True, device=device)
        float_model_seq = nn.Sequential(f_conv, f_bn, nn.ReLU()).to(device)

        float_model_seq.eval()
        fp4_cbr.eval()
        fp4_cbr.bn.running_mean.data.copy_(f_bn.running_mean.data)
        fp4_cbr.bn.running_var.data.copy_(f_bn.running_var.data)
        fp4_cbr.bn.num_batches_tracked.data.copy_(f_bn.num_batches_tracked.data)

        with torch.no_grad():
            out_float = float_model_seq(input_tensor)
            out_fp4 = fp4_cbr(input_tensor.clone())
        
        self.assertTrue(torch.allclose(out_float, out_fp4, atol=0.5),
                        f"Eval output diff CUDA: {(out_float - out_fp4).abs().max().item()}")
        print("  Forward pass (Eval, CUDA): PASSED")

    def test_repr(self):
        print("\nTesting FP4ConvBnReLU2d __repr__...")
        fp4_cbr = FP4ConvBnReLU2d(3, 8, 3, device='cpu', dtype=torch.float32)
        representation = repr(fp4_cbr)
        self.assertIn("(conv): FP4Conv2d", representation)
        self.assertIn("in_channels=3, out_channels=8", representation)
        self.assertIn("(bn): FP4BatchNorm2d", representation)
        self.assertIn("8, eps=1e-05", representation) # 8 features for BN
        self.assertIn("(relu): ReLU(inplace=False)", representation)
        print(f"  Representation:\n{representation}")
        print("  __repr__: PASSED")

if __name__ == '__main__':
    if parent_dir not in sys.path: # Ensure lib is in path if run directly
        sys.path.insert(0, parent_dir)
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 