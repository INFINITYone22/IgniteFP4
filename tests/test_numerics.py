# tests/test_numerics.py
"""
Unit tests for the FP4 numerical functions in `ignitefp4_lib.numerics`.

This test suite covers:
- Symmetric quantization:
  - Scale calculation (`calculate_symmetric_scale`).
  - Quantization to FP4 (`quantize_to_fp4_symmetric`).
  - Dequantization from FP4 (`dequantize_from_fp4_symmetric`).
  - Roundtrip accuracy (quantize then dequantize).
- Asymmetric quantization:
  - Scale and zero-point calculation (`calculate_asymmetric_scale_zeropoint`).
  - Quantization to FP4 (`quantize_to_fp4_asymmetric`).
  - Dequantization from FP4 (`dequantize_from_fp4_asymmetric`).
  - Roundtrip accuracy.

Tests include edge cases like all-zero tensors, constant value tensors,
and tensors with values that span positive and negative ranges.
"""
import unittest
import torch
import sys
import os

# Adjust path to import from ignitefp4_lib assuming tests is a sibling directory
# This is a common way to handle imports for local package testing.
# More robust solutions involve proper packaging or using pytest with src layout.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ignitefp4_lib.numerics import (
    calculate_symmetric_scale,
    quantize_to_fp4_symmetric,
    dequantize_from_fp4_symmetric,
    FP4_SIGNED_SYMMETRIC_MIN_VAL,
    FP4_SIGNED_SYMMETRIC_MAX_VAL,
    calculate_asymmetric_scale_zeropoint,
    quantize_to_fp4_asymmetric,
    dequantize_from_fp4_asymmetric,
    FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL,
    FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL
)

class TestFP4Numerics(unittest.TestCase):
    """Test cases for FP4 numerical utility functions."""

    def test_calculate_symmetric_scale(self):
        """Tests the calculation of scale for symmetric quantization."""
        print("\nTesting calculate_symmetric_scale...")
        tensor1 = torch.tensor([-10.0, 0.0, 5.0, 14.0]) 
        scale1 = calculate_symmetric_scale(tensor1)
        self.assertAlmostEqual(scale1.item(), 2.0, places=5)
        print(f"  tensor1 scale: {scale1.item()} (Expected: ~2.0)")

        tensor2 = torch.tensor([0.0, 0.0, 0.0])
        scale2 = calculate_symmetric_scale(tensor2)
        self.assertEqual(scale2.item(), 1.0)
        print(f"  tensor2 (all zeros) scale: {scale2.item()} (Expected: 1.0)")

        tensor3 = torch.tensor([-7.0, -3.5, -1.0]) 
        scale3 = calculate_symmetric_scale(tensor3)
        self.assertAlmostEqual(scale3.item(), 1.0, places=5)
        print(f"  tensor3 scale: {scale3.item()} (Expected: ~1.0)")

        tensor4 = torch.tensor([0.0001, -0.00005]) 
        scale4 = calculate_symmetric_scale(tensor4)
        self.assertAlmostEqual(scale4.item(), 0.0001 / 7.0, places=8)
        print(f"  tensor4 scale: {scale4.item()} (Expected: ~{0.0001 / 7.0})")
        
        tensor5 = torch.tensor([-3.5]) 
        scale5 = calculate_symmetric_scale(tensor5)
        self.assertAlmostEqual(scale5.item(), 0.5, places=5)
        print(f"  tensor5 scale: {scale5.item()} (Expected: ~0.5)")

    def test_quantize_to_fp4_symmetric(self):
        """Tests quantization to signed symmetric FP4 representation."""
        print("\nTesting quantize_to_fp4_symmetric...")
        tensor1 = torch.tensor([-14.0, -7.0, 0.0, 7.0, 14.0])
        quant_tensor1, scale1 = quantize_to_fp4_symmetric(tensor1)
        expected_quant1 = torch.tensor([-7, -4, 0, 4, 7], dtype=torch.int8)
        self.assertTrue(torch.equal(quant_tensor1, expected_quant1))
        self.assertAlmostEqual(scale1.item(), 2.0, places=5)
        print(f"  tensor1 quantized: {quant_tensor1.tolist()}, scale: {scale1.item()}")

        tensor_bound_test = torch.tensor([-16.0, -8.0, 0.0, 8.0, 16.0])
        quant_tensor_bound, scale_bound = quantize_to_fp4_symmetric(tensor_bound_test)
        expected_quant_bound = torch.tensor([-7, -4, 0, 4, 7], dtype=torch.int8)
        self.assertTrue(torch.equal(quant_tensor_bound, expected_quant_bound))
        print(f"  tensor_bound_test quantized: {quant_tensor_bound.tolist()}, scale: {scale_bound.item()}")

        tensor3 = torch.zeros((2,2))
        quant_tensor3, scale3 = quantize_to_fp4_symmetric(tensor3)
        self.assertTrue(torch.equal(quant_tensor3, torch.zeros((2,2), dtype=torch.int8)))
        self.assertEqual(scale3.item(), 1.0)
        print(f"  tensor3 (all_zeros) quantized: {quant_tensor3.tolist()}, scale: {scale3.item()}")

    def test_dequantize_from_fp4_symmetric(self):
        """Tests dequantization from signed symmetric FP4 representation."""
        print("\nTesting dequantize_from_fp4_symmetric...")
        quant_tensor1 = torch.tensor([-7, -4, 0, 4, 7], dtype=torch.int8)
        scale1 = torch.tensor(2.0)
        dequant_tensor1 = dequantize_from_fp4_symmetric(quant_tensor1, scale1)
        expected_float1 = torch.tensor([-14.0, -8.0, 0.0, 8.0, 14.0])
        self.assertTrue(torch.allclose(dequant_tensor1, expected_float1, atol=1e-5))
        print(f"  quant_tensor1 dequantized: {dequant_tensor1.tolist()}")

        quant_tensor2 = torch.tensor([-7, -4, 0, 4, 7], dtype=torch.int8)
        scale2 = torch.tensor(16.0/7.0)
        dequant_tensor2 = dequantize_from_fp4_symmetric(quant_tensor2, scale2)
        expected_float2 = torch.tensor([-16.0, -64.0/7.0, 0.0, 64.0/7.0, 16.0])
        self.assertTrue(torch.allclose(dequant_tensor2, expected_float2, atol=1e-5))
        print(f"  quant_tensor2 dequantized: {dequant_tensor2.tolist()}")

    def test_quant_dequant_roundtrip(self):
        """Tests the roundtrip consistency of symmetric quantization and dequantization."""
        print("\nTesting quant_dequant_roundtrip...")
        original_tensor1 = torch.randn((5, 5)) * 20
        quant_t, scale_t = quantize_to_fp4_symmetric(original_tensor1)
        dequant_t = dequantize_from_fp4_symmetric(quant_t, scale_t)
        
        self.assertTrue(quant_t.max() <= FP4_SIGNED_SYMMETRIC_MAX_VAL)
        self.assertTrue(quant_t.min() >= FP4_SIGNED_SYMMETRIC_MIN_VAL)
        
        max_abs_error = torch.max(torch.abs(original_tensor1 - dequant_t))
        self.assertTrue(max_abs_error <= (scale_t / 2.0) + 1e-6,
                        f"Max abs error {max_abs_error.item()} > scale/2 { (scale_t / 2.0).item()}")
        print(f"  original_tensor1 (sample): {original_tensor1[0,:3].tolist()}")
        print(f"  quantized_tensor1 (sample): {quant_t[0,:3].tolist()}, scale: {scale_t.item()}")
        print(f"  dequantized_tensor1 (sample): {dequant_t[0,:3].tolist()}")
        print(f"  Max absolute error: {max_abs_error.item()}, Expected max error (scale/2): {(scale_t/2.0).item()}")

        original_tensor2 = torch.tensor([0.0, 0.0, 0.1, -0.2, 0.0, 1.0, -0.5, 0.0])
        quant_t2, scale_t2 = quantize_to_fp4_symmetric(original_tensor2)
        dequant_t2 = dequantize_from_fp4_symmetric(quant_t2, scale_t2)
        max_abs_error2 = torch.max(torch.abs(original_tensor2 - dequant_t2))
        self.assertTrue(quant_t2.max() <= FP4_SIGNED_SYMMETRIC_MAX_VAL)
        self.assertTrue(quant_t2.min() >= FP4_SIGNED_SYMMETRIC_MIN_VAL)
        self.assertTrue(max_abs_error2 <= (scale_t2 / 2.0) + 1e-6,
                        f"Max abs error {max_abs_error2.item()} > scale/2 { (scale_t2 / 2.0).item()}")
        print(f"  original_tensor2: {original_tensor2.tolist()}")
        print(f"  quantized_tensor2: {quant_t2.tolist()}, scale: {scale_t2.item()}")
        print(f"  dequantized_tensor2: {dequant_t2.tolist()}")
        print(f"  Max absolute error for tensor2: {max_abs_error2.item()}, Expected max error (scale/2): {(scale_t2/2.0).item()}")
        
        original_tensor3 = torch.tensor([-14.0, -10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5, 14.0])
        quant_t3, scale_t3 = quantize_to_fp4_symmetric(original_tensor3)
        dequant_t3 = dequantize_from_fp4_symmetric(quant_t3, scale_t3)
        print(f"  original_tensor3: {original_tensor3.tolist()}")
        print(f"  quant_t3: {quant_t3.tolist()}, scale_t3: {scale_t3.item()} (Expected scale: 2.0)")
        print(f"  dequant_t3: {dequant_t3.tolist()}")
        max_abs_error3 = torch.max(torch.abs(original_tensor3 - dequant_t3))
        self.assertTrue(max_abs_error3 <= (scale_t3 / 2.0) + 1e-6, 
                        f"Max abs error {max_abs_error3.item()} > scale/2 { (scale_t3.item()/2.0)}")

    def test_asymmetric_quantization(self):
        """Tests the full workflow of asymmetric quantization: scale/zp calculation, quantize, dequantize."""
        print("\nTesting asymmetric_quantization...")
        # Imports are fine at the top of the file, re-importing here for clarity of what's new
        from ignitefp4_lib.numerics import (
            calculate_asymmetric_scale_zeropoint,
            quantize_to_fp4_asymmetric,
            dequantize_from_fp4_asymmetric,
            FP4_UNSIGNED_ASYMMETRIC_MIN_QUANT_VAL as QMIN,
            FP4_UNSIGNED_ASYMMETRIC_MAX_QUANT_VAL as QMAX
        )

        # 1. Basic test: positive values, min_val is 0
        tensor1 = torch.tensor([0.0, 1.0, 7.5, 15.0, 20.0]) # min=0, max=20
        scale1, zp1 = calculate_asymmetric_scale_zeropoint(tensor1)
        self.assertAlmostEqual(scale1.item(), 20.0/15.0, places=5) # (max-min)/(QMAX-QMIN)
        self.assertEqual(zp1.item(), QMIN) # If min_val is 0, zero_point should ideally be QMIN
        print(f"  tensor1 (positive, zero_min): scale={scale1.item():.4f}, zp={zp1.item()}")
        
        quant_t1 = quantize_to_fp4_asymmetric(tensor1, scale1, zp1)
        # X_q = round(X_f / scale + zero_point)
        # round(0.0 / (20/15) + 0) = 0
        # round(1.0 / (20/15) + 0) = round(0.75) = 1
        # round(7.5 / (20/15) + 0) = round(5.625) = 6
        # round(15.0 / (20/15) + 0) = round(11.25) = 11
        # round(20.0 / (20/15) + 0) = round(15.0) = 15
        expected_q1 = torch.tensor([0, 1, 6, 11, 15], dtype=torch.int8)
        self.assertTrue(torch.equal(quant_t1, expected_q1))
        dequant_t1 = dequantize_from_fp4_asymmetric(quant_t1, scale1, zp1)
        # (X_q - zp) * scale
        expected_dequant1_values = (expected_q1.float() - zp1.float()) * scale1
        self.assertTrue(torch.allclose(dequant_t1, expected_dequant1_values, atol=1e-5))

        # 2. Test with range not starting at 0
        tensor2 = torch.tensor([5.0, 10.0, 12.5, 15.0, 20.0]) # min=5, max=20
        scale2, zp2 = calculate_asymmetric_scale_zeropoint(tensor2)
        # scale = (20-5)/(15-0) = 1.0
        # zp_float = QMIN - min_val/scale = 0 - 5.0/1.0 = -5.0. round(-5)=-5. clamp(-5,0,15)=0
        self.assertAlmostEqual(scale2.item(), 1.0, places=5)
        self.assertEqual(zp2.item(), 0) 
        print(f"  tensor2 (offset positive): scale={scale2.item():.4f}, zp={zp2.item()}")
        quant_t2 = quantize_to_fp4_asymmetric(tensor2, scale2, zp2)
        # X_q = round(X_f / 1.0 + 0)
        expected_q2 = torch.tensor([5, 10, 13, 15, 15], dtype=torch.int8) # round(12.5)=13, round(20)=20->clamped 15
        self.assertTrue(torch.equal(quant_t2, expected_q2))
        dequant_t2 = dequantize_from_fp4_asymmetric(quant_t2, scale2, zp2)
        expected_dequant2_values = (expected_q2.float() - zp2.float()) * scale2
        self.assertTrue(torch.allclose(dequant_t2, expected_dequant2_values, atol=1e-5))

        # 3. All zeros tensor
        tensor3 = torch.zeros((2,2))
        scale3, zp3 = calculate_asymmetric_scale_zeropoint(tensor3)
        self.assertEqual(scale3.item(), 1.0) # Default scale for flat tensor if min_val==0
        self.assertEqual(zp3.item(), 0)      # Default ZP for flat tensor if min_val==0
        quant_t3 = quantize_to_fp4_asymmetric(tensor3, scale3, zp3)
        self.assertTrue(torch.equal(quant_t3, torch.zeros((2,2), dtype=torch.int8)))
        print(f"  tensor3 (all zeros): scale={scale3.item()}, zp={zp3.item()}")

        # 4. Constant non-zero tensor
        tensor4 = torch.ones((2,2)) * 5.0 # min=5, max=5
        scale4, zp4 = calculate_asymmetric_scale_zeropoint(tensor4)
        # scale should be ~eps, zp should be (QMIN+QMAX)//2 = 7
        self.assertTrue(scale4.item() < 1e-5) # Check scale is very small
        self.assertEqual(zp4.item(), (QMIN + QMAX) // 2)
        quant_t4 = quantize_to_fp4_asymmetric(tensor4, scale4, zp4)
        # All values should quantize to the zero_point for this case
        expected_q4 = torch.ones_like(quant_t4) * zp4.item()
        self.assertTrue(torch.equal(quant_t4, expected_q4))
        print(f"  tensor4 (constant 5.0): scale={scale4.item():.2e}, zp={zp4.item()}")
        
        # 5. Roundtrip test with random positive data
        original_tensor5 = torch.rand((3,3)) * 10 + 2 # Values between ~2 and ~12
        scale5, zp5 = calculate_asymmetric_scale_zeropoint(original_tensor5)
        quant_t5 = quantize_to_fp4_asymmetric(original_tensor5, scale5, zp5)
        dequant_t5 = dequantize_from_fp4_asymmetric(quant_t5, scale5, zp5)

        self.assertTrue(quant_t5.max().item() <= QMAX)
        self.assertTrue(quant_t5.min().item() >= QMIN)
        
        max_abs_error = torch.max(torch.abs(original_tensor5 - dequant_t5))
        # Max quantization error for asymmetric is typically scale / 2
        self.assertTrue(max_abs_error.item() <= (scale5.item() / 2.0) + 1e-5, 
                        f"Roundtrip Max abs error {max_abs_error.item()} > scale/2 { (scale5.item() / 2.0)}")
        print(f"  tensor5 roundtrip: scale={scale5.item():.4f}, zp={zp5.item()}, max_err={max_abs_error.item():.4f}")

        # 6. Test with negative values included - asymmetric unsigned should map them near 0
        tensor6 = torch.tensor([-10.0, -1.0, 0.0, 5.0, 10.0]) # min=-10, max=10
        scale6, zp6 = calculate_asymmetric_scale_zeropoint(tensor6)
        # scale = (10 - (-10)) / 15 = 20/15 = 4/3
        # zp_float = 0 - (-10 / (4/3)) = 0 - (-7.5) = 7.5. round(7.5)=8. clamp(8,0,15)=8
        self.assertAlmostEqual(scale6.item(), 20.0/15.0, places=5)
        self.assertEqual(zp6.item(), 8)
        print(f"  tensor6 (neg included): scale={scale6.item():.4f}, zp={zp6.item()}")

        quant_t6 = quantize_to_fp4_asymmetric(tensor6, scale6, zp6)
        # X_q = round(X_f / scale + zero_point)
        # round(-10/(4/3)+8) = round(-7.5+8) = round(0.5) = 1
        # round(-1/(4/3)+8) = round(-0.75+8) = round(7.25) = 7
        # round(0/(4/3)+8) = 8
        # round(5/(4/3)+8) = round(3.75+8) = round(11.75) = 12
        # round(10/(4/3)+8) = round(7.5+8) = round(15.5) = 16 -> clamped to 15
        expected_q6 = torch.tensor([1, 7, 8, 12, 15], dtype=torch.int8)
        self.assertTrue(torch.equal(quant_t6, expected_q6))
        dequant_t6 = dequantize_from_fp4_asymmetric(quant_t6, scale6, zp6)
        expected_dequant6_values = (expected_q6.float() - zp6.float()) * scale6
        self.assertTrue(torch.allclose(dequant_t6, expected_dequant6_values, atol=1e-5))

if __name__ == '__main__':
    # This allows running the test script directly.
    # Adding a check to ensure the parent directory is in sys.path if not already.
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 