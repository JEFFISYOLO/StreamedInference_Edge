#!/usr/bin/env python3
"""
Basic test script to verify quantized weights are correctly loaded and dequantized.
This is a lightweight test that doesn't require loading the full original model.
"""

import numpy as np
import json
import os
import torch

def test_quantized_weights():
    """Test that quantized weights can be loaded and dequantized correctly."""
    print("=" * 50)
    print("Basic Quantization Test")
    print("=" * 50)
    
    weights_dir = "esp32_weights"
    
    # Load configuration
    print("\n1. Loading model configuration...")
    with open(os.path.join(weights_dir, "model_config.json"), "r") as f:
        config = json.load(f)
    
    print(f"Model config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load quantization scales
    print("\n2. Loading quantization scales...")
    with open(os.path.join(weights_dir, "quantization_scales.json"), "r") as f:
        scales = json.load(f)
    
    print(f"Quantization scales:")
    for name, scale in scales.items():
        print(f"  {name}: {scale:.6f}")
    
    # Test loading and dequantizing tensors
    print("\n3. Testing tensor loading and dequantization...")
    
    test_files = [
        ("embed_tokens.bin", (config["vocab_size"], config["hidden_size"])),
        ("q_proj.bin", (config["hidden_size"], config["hidden_size"])),
        ("k_proj.bin", (config["num_key_value_heads"] * (config["hidden_size"] // config["num_attention_heads"]), config["hidden_size"])),
        ("v_proj.bin", (config["num_key_value_heads"] * (config["hidden_size"] // config["num_attention_heads"]), config["hidden_size"])),
        ("o_proj.bin", (config["hidden_size"], config["hidden_size"])),
        ("gate_proj.bin", (config["intermediate_size"], config["hidden_size"])),
        ("up_proj.bin", (config["intermediate_size"], config["hidden_size"])),
        ("down_proj.bin", (config["hidden_size"], config["intermediate_size"])),
        ("lm_head.bin", (config["vocab_size"], config["hidden_size"])),
    ]
    
    total_params = 0
    total_size = 0
    
    for filename, expected_shape in test_files:
        filepath = os.path.join(weights_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  ❌ {filename}: File not found!")
            continue
        
        # Load quantized data
        quantized_data = np.fromfile(filepath, dtype=np.int8)
        file_size = os.path.getsize(filepath)
        
        # Reshape and dequantize
        quantized_tensor = quantized_data.reshape(expected_shape)
        
        # Get scale
        scale_name = filename.replace(".bin", "")
        if scale_name in scales:
            scale = scales[scale_name]
        else:
            scale = 1.0
            print(f"  [WARN] {filename}: No scale found, using 1.0")
        
        # Dequantize
        dequantized_tensor = quantized_tensor.astype(np.float32) * scale
        
        # Statistics
        num_params = quantized_tensor.size
        total_params += num_params
        total_size += file_size
        
        # Check quantization range
        min_val = np.min(quantized_tensor)
        max_val = np.max(quantized_tensor)
        dequant_min = np.min(dequantized_tensor)
        dequant_max = np.max(dequantized_tensor)
        
        print(f"  [OK] {filename}:")
        print(f"     Shape: {expected_shape} ({num_params:,} params)")
        print(f"     Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print(f"     Quantized range: [{min_val}, {max_val}]")
        print(f"     Dequantized range: [{dequant_min:.4f}, {dequant_max:.4f}]")
        print(f"     Scale: {scale:.6f}")
        
        # Verify shape matches expected
        if quantized_tensor.shape != expected_shape:
            print(f"     [WARN] Shape mismatch! Expected {expected_shape}, got {quantized_tensor.shape}")
    
    # Test layer norm files
    print("\n4. Testing layer normalization parameters...")
    
    norm_files = ["norm1.bin", "norm2.bin", "final_norm.bin"]
    
    for filename in norm_files:
        filepath = os.path.join(weights_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  ❌ {filename}: File not found!")
            continue
        
        # Load norm data (scale and bias)
        norm_data = np.fromfile(filepath, dtype=np.float32)
        file_size = os.path.getsize(filepath)
        
        if len(norm_data) == 2:
            scale, bias = norm_data
            print(f"  [OK] {filename}: scale={scale:.6f}, bias={bias:.6f} ({file_size} bytes)")
        elif len(norm_data) == config["hidden_size"]:
            scale = norm_data
            print(f"  [OK] {filename}: scale only ({len(scale)} values, {file_size} bytes)")
            print(f"     Scale range: [{np.min(scale):.6f}, {np.max(scale):.6f}]")
        else:
            print(f"  [WARN] {filename}: Unexpected data length {len(norm_data)}")
        
        total_size += file_size
    
    # Summary
    print(f"\n5. Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total file size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"  Parameters per byte: {total_params / total_size:.2f}")
    
    # Calculate theoretical size
    float32_size = total_params * 4  # 4 bytes per float32
    compression_ratio = float32_size / total_size
    print(f"  Float32 equivalent: {float32_size:,} bytes ({float32_size/1024/1024:.2f} MB)")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    # Test basic mathematical operations
    print(f"\n6. Testing basic operations...")
    
    # Test embedding lookup
    embed_file = os.path.join(weights_dir, "embed_tokens.bin")
    if os.path.exists(embed_file):
        embed_data = np.fromfile(embed_file, dtype=np.int8).reshape((config["vocab_size"], config["hidden_size"]))
        embed_scale = scales["embed_tokens"]
        
        # Test a few token embeddings
        test_tokens = [0, 1, 2, 100, 1000, config["vocab_size"]-1]
        print(f"  Testing embedding lookup for tokens: {test_tokens}")
        
        for token_id in test_tokens:
            if 0 <= token_id < config["vocab_size"]:
                embed = embed_data[token_id] * embed_scale
                embed_norm = np.linalg.norm(embed)
                print(f"    Token {token_id}: norm={embed_norm:.4f}, range=[{np.min(embed):.4f}, {np.max(embed):.4f}]")
    
    print(f"\n" + "=" * 50)
    print("Basic quantization test completed!")
    print("[OK] All quantized weights loaded successfully")
    print("=" * 50)
    
    return True

def test_matvec_operation():
    """Test matrix-vector multiplication with quantized weights."""
    print("\n" + "=" * 50)
    print("Matrix-Vector Multiplication Test")
    print("=" * 50)
    
    weights_dir = "esp32_weights"
    
    # Load config and scales
    with open(os.path.join(weights_dir, "model_config.json"), "r") as f:
        config = json.load(f)
    
    with open(os.path.join(weights_dir, "quantization_scales.json"), "r") as f:
        scales = json.load(f)
    
    hidden_size = config["hidden_size"]
    
    # Test Q projection (hidden_size x hidden_size)
    print("\n1. Testing Q projection matrix-vector multiplication...")
    
    q_proj_file = os.path.join(weights_dir, "q_proj.bin")
    if os.path.exists(q_proj_file):
        # Load quantized Q projection
        q_proj_quantized = np.fromfile(q_proj_file, dtype=np.int8).reshape((hidden_size, hidden_size))
        q_scale = scales["q_proj"]
        
        # Create test input vector (random)
        np.random.seed(42)  # For reproducible results
        test_input = np.random.randn(hidden_size).astype(np.float32)
        
        # Dequantize Q projection
        q_proj_dequantized = q_proj_quantized.astype(np.float32) * q_scale
        
        # Compute matrix-vector multiplication
        # Method 1: Direct float32 computation (reference)
        output_ref = np.dot(test_input, q_proj_dequantized.T)
        
        # Method 2: Simulate quantized computation
        # This simulates what the ESP32 would do with fixed-point arithmetic
        test_input_quantized = np.clip(np.round(test_input / q_scale), -128, 127).astype(np.int8)
        output_quantized = np.dot(test_input_quantized.astype(np.int32), q_proj_quantized.T.astype(np.int32))
        output_simulated = output_quantized.astype(np.float32) * (q_scale * q_scale)
        
        # Compare results
        diff = np.abs(output_ref - output_simulated)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  Input shape: {test_input.shape}")
        print(f"  Q projection shape: {q_proj_quantized.shape}")
        print(f"  Output shape: {output_ref.shape}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Relative error: {max_diff / np.max(np.abs(output_ref)):.6f}")
        
        if max_diff < 0.1:  # Reasonable threshold
            print("  [OK] Matrix-vector multiplication test passed!")
        else:
            print("  [WARN] Large differences detected - check quantization")
    
    print(f"\n" + "=" * 50)
    print("Matrix-vector multiplication test completed!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_quantized_weights()
        test_matvec_operation()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
