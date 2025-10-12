#!/usr/bin/env python3
"""
Quantize Tiny-LLM model to int8 and prepare for ESP32 streaming inference.
This script converts the model weights to int8 format and organizes them
for efficient streaming from SD card during inference.
"""

import torch
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

def quantize_tensor(tensor, bits=8):
    """Quantize tensor to int8 with symmetric quantization."""
    # Calculate quantization parameters
    scale = tensor.abs().max() / (2**(bits-1) - 1)
    if scale == 0:
        scale = 1.0
    
    # Quantize to int8 range [-128, 127]
    quantized = torch.clamp(torch.round(tensor / scale), -128, 127)
    return quantized.to(torch.int8), scale

def save_tensor_binary(tensor, filename):
    """Save tensor as binary file for ESP32."""
    # Convert to numpy and ensure correct byte order
    np_array = tensor.detach().cpu().numpy()
    np_array.astype(np.int8).tofile(filename)
    print(f"Saved {tensor.shape} tensor to {filename} ({np_array.nbytes} bytes)")

def save_scale_bias(scale, bias, filename):
    """Save scale and bias as float32 binary for ESP32."""
    if bias is not None:
        combined = torch.stack([scale, bias])
    else:
        combined = scale.unsqueeze(0)
    combined.detach().cpu().numpy().astype(np.float32).tofile(filename)
    print(f"Saved scale/bias to {filename} ({combined.numel() * 4} bytes)")

def main():
    MODEL_NAME = "arnir0/Tiny-LLM"
    OUTPUT_DIR = "esp32_weights"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Tiny-LLM model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Save tokenizer for ESP32
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Get model configuration
    config = model.config
    print(f"Model config: {config}")
    
    # Save model config for ESP32
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
        json.dump({
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "hidden_act": config.hidden_act,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
        }, f, indent=2)
    
    print("\nQuantizing model weights...")
    
    # 1. Input embeddings (vocab_size, hidden_size)
    embed_tokens = model.model.embed_tokens.weight
    embed_quantized, embed_scale = quantize_tensor(embed_tokens)
    save_tensor_binary(embed_quantized, os.path.join(OUTPUT_DIR, "embed_tokens.bin"))
    
    # 2. Layer 0 components
    layer = model.model.layers[0]
    
    # Layer norm 1 (RMSNorm)
    norm1_scale = layer.input_layernorm.weight
    norm1_bias = getattr(layer.input_layernorm, 'bias', None)
    save_scale_bias(norm1_scale, norm1_bias, os.path.join(OUTPUT_DIR, "norm1.bin"))
    
    # Self-attention projections
    # Q projection (hidden_size, hidden_size) - full projection for 2 heads
    q_proj = layer.self_attn.q_proj.weight
    q_quantized, q_scale = quantize_tensor(q_proj)
    save_tensor_binary(q_quantized, os.path.join(OUTPUT_DIR, "q_proj.bin"))
    
    # K projection (num_kv_heads * head_dim, hidden_size) - 1 KV head
    k_proj = layer.self_attn.k_proj.weight
    k_quantized, k_scale = quantize_tensor(k_proj)
    save_tensor_binary(k_quantized, os.path.join(OUTPUT_DIR, "k_proj.bin"))
    
    # V projection (num_kv_heads * head_dim, hidden_size) - 1 KV head
    v_proj = layer.self_attn.v_proj.weight
    v_quantized, v_scale = quantize_tensor(v_proj)
    save_tensor_binary(v_quantized, os.path.join(OUTPUT_DIR, "v_proj.bin"))
    
    # Output projection (hidden_size, hidden_size)
    o_proj = layer.self_attn.o_proj.weight
    o_quantized, o_scale = quantize_tensor(o_proj)
    save_tensor_binary(o_quantized, os.path.join(OUTPUT_DIR, "o_proj.bin"))
    
    # Layer norm 2 (RMSNorm)
    norm2_scale = layer.post_attention_layernorm.weight
    norm2_bias = getattr(layer.post_attention_layernorm, 'bias', None)
    save_scale_bias(norm2_scale, norm2_bias, os.path.join(OUTPUT_DIR, "norm2.bin"))
    
    # MLP projections (SwiGLU)
    # Gate projection (intermediate_size, hidden_size)
    gate_proj = layer.mlp.gate_proj.weight
    gate_quantized, gate_scale = quantize_tensor(gate_proj)
    save_tensor_binary(gate_quantized, os.path.join(OUTPUT_DIR, "gate_proj.bin"))
    
    # Up projection (intermediate_size, hidden_size)
    up_proj = layer.mlp.up_proj.weight
    up_quantized, up_scale = quantize_tensor(up_proj)
    save_tensor_binary(up_quantized, os.path.join(OUTPUT_DIR, "up_proj.bin"))
    
    # Down projection (hidden_size, intermediate_size)
    down_proj = layer.mlp.down_proj.weight
    down_quantized, down_scale = quantize_tensor(down_proj)
    save_tensor_binary(down_quantized, os.path.join(OUTPUT_DIR, "down_proj.bin"))
    
    # 3. Final layer norm (RMSNorm)
    final_norm_scale = model.model.norm.weight
    final_norm_bias = getattr(model.model.norm, 'bias', None)
    save_scale_bias(final_norm_scale, final_norm_bias, os.path.join(OUTPUT_DIR, "final_norm.bin"))
    
    # 4. LM head (vocab_size, hidden_size)
    lm_head = model.lm_head.weight
    lm_quantized, lm_scale = quantize_tensor(lm_head)
    save_tensor_binary(lm_quantized, os.path.join(OUTPUT_DIR, "lm_head.bin"))
    
    # Save quantization scales for ESP32
    scales = {
        "embed_tokens": float(embed_scale),
        "q_proj": float(q_scale),
        "k_proj": float(k_scale),
        "v_proj": float(v_scale),
        "o_proj": float(o_scale),
        "gate_proj": float(gate_scale),
        "up_proj": float(up_scale),
        "down_proj": float(down_scale),
        "lm_head": float(lm_scale),
    }
    
    with open(os.path.join(OUTPUT_DIR, "quantization_scales.json"), "w") as f:
        json.dump(scales, f, indent=2)
    
    print(f"\nQuantization complete! Files saved to {OUTPUT_DIR}/")
    print("\nFile sizes:")
    total_size = 0
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('.bin'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, file))
            total_size += size
            print(f"  {file}: {size:,} bytes")
    
    print(f"\nTotal quantized model size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"Original model size (estimated): {sum(p.numel() for p in model.parameters()) * 4:,} bytes ({sum(p.numel() for p in model.parameters()) * 4/1024/1024:.2f} MB)")
    print(f"Compression ratio: {sum(p.numel() for p in model.parameters()) * 4 / total_size:.1f}x")

if __name__ == "__main__":
    main()

