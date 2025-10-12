#!/usr/bin/env python3
"""
Test script to verify float32 weights are correctly saved and can be loaded.
This is a quick validation before flashing to ESP32.
"""

import numpy as np
import json
import os

def test_float32_weights():
    """Test that float32 weights can be loaded correctly."""
    print("=" * 50)
    print("Float32 Weights Validation Test")
    print("=" * 50)
    
    weights_dir = "esp32_float32_weights"
    
    if not os.path.exists(weights_dir):
        print(f"[ERROR] Directory {weights_dir} not found!")
        print("Run prepare_float32_weights.py first to generate weights.")
        return False
    
    # Load configuration
    print("\n1. Loading model configuration...")
    config_file = os.path.join(weights_dir, "model_config.json")
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file {config_file} not found!")
        return False
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    print(f"Model config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test loading and validation of tensors
    print("\n2. Testing tensor loading and validation...")
    
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
    all_good = True
    
    for filename, expected_shape in test_files:
        filepath = os.path.join(weights_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  [ERROR] {filename}: File not found!")
            all_good = False
            continue
        
        # Load float32 data
        float_data = np.fromfile(filepath, dtype=np.float32)
        file_size = os.path.getsize(filepath)
        
        # Reshape and validate
        try:
            float_tensor = float_data.reshape(expected_shape)
        except ValueError as e:
            print(f"  [ERROR] {filename}: Shape mismatch! Expected {expected_shape}, got {float_data.shape}")
            print(f"    Error: {e}")
            all_good = False
            continue
        
        # Statistics
        num_params = float_tensor.size
        total_params += num_params
        total_size += file_size
        
        # Check value ranges
        min_val = np.min(float_tensor)
        max_val = np.max(float_tensor)
        mean_val = np.mean(float_tensor)
        std_val = np.std(float_tensor)
        
        print(f"  [OK] {filename}:")
        print(f"     Shape: {expected_shape} ({num_params:,} params)")
        print(f"     Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"     Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"     Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        
        # Check for reasonable values (not all zeros or NaNs)
        if np.all(float_tensor == 0):
            print(f"     [WARN] All values are zero!")
        if np.any(np.isnan(float_tensor)):
            print(f"     [ERROR] Contains NaN values!")
            all_good = False
        if np.any(np.isinf(float_tensor)):
            print(f"     [ERROR] Contains infinite values!")
            all_good = False
    
    # Test layer norm files
    print("\n3. Testing layer normalization parameters...")
    
    norm_files = ["norm1.bin", "norm2.bin", "final_norm.bin"]
    
    for filename in norm_files:
        filepath = os.path.join(weights_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  [ERROR] {filename}: File not found!")
            all_good = False
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
    print(f"\n4. Summary:")
    if all_good:
        print(f"  [SUCCESS] All float32 weights validated successfully!")
    else:
        print(f"  [ERROR] Some validation errors found!")
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total file size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"  Parameters per byte: {total_params / total_size:.2f}")
    
    # Calculate theoretical size
    float32_size = total_params * 4  # 4 bytes per float32
    print(f"  Float32 equivalent: {float32_size:,} bytes ({float32_size/1024/1024:.2f} MB)")
    
    # Test basic mathematical operations
    print(f"\n5. Testing basic operations...")
    
    # Test embedding lookup
    embed_file = os.path.join(weights_dir, "embed_tokens.bin")
    if os.path.exists(embed_file):
        embed_data = np.fromfile(embed_file, dtype=np.float32).reshape((config["vocab_size"], config["hidden_size"]))
        
        # Test a few token embeddings
        test_tokens = [0, 1, 2, 100, 1000, config["vocab_size"]-1]
        print(f"  Testing embedding lookup for tokens: {test_tokens}")
        
        for token_id in test_tokens:
            if 0 <= token_id < config["vocab_size"]:
                embed = embed_data[token_id]
                embed_norm = np.linalg.norm(embed)
                print(f"    Token {token_id}: norm={embed_norm:.4f}, range=[{np.min(embed):.4f}, {np.max(embed):.4f}]")
    
    # Test matrix-vector multiplication
    print(f"\n6. Testing matrix-vector operations...")
    
    # Test Q projection (hidden_size x hidden_size)
    q_proj_file = os.path.join(weights_dir, "q_proj.bin")
    if os.path.exists(q_proj_file):
        hidden_size = config["hidden_size"]
        q_proj = np.fromfile(q_proj_file, dtype=np.float32).reshape((hidden_size, hidden_size))
        
        # Create test input vector
        np.random.seed(42)
        test_input = np.random.randn(hidden_size).astype(np.float32)
        
        # Compute matrix-vector multiplication
        output = np.dot(test_input, q_proj.T)
        
        print(f"  Q projection test:")
        print(f"    Input shape: {test_input.shape}")
        print(f"    Q projection shape: {q_proj.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Output range: [{np.min(output):.4f}, {np.max(output):.4f}]")
        print(f"    Output norm: {np.linalg.norm(output):.4f}")
    
    print(f"\n" + "=" * 50)
    if all_good:
        print("Float32 weights validation completed successfully!")
        print("Ready for ESP32 deployment! 🚀")
    else:
        print("Float32 weights validation failed!")
        print("Check errors above before deploying to ESP32.")
    print("=" * 50)
    
    return all_good

if __name__ == "__main__":
    try:
        test_float32_weights()
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
