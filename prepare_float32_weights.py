#!/usr/bin/env python3
"""
Prepare Tiny-LLM model weights in float32 format for ESP32-CAM streaming inference.
This script extracts the original float32 weights and saves them in a format
suitable for streaming from SD card without quantization.
"""

import torch
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_tensor_binary(tensor, filename):
    """Save tensor as binary file for ESP32."""
    # Convert to numpy and ensure correct byte order
    np_array = tensor.detach().cpu().numpy()
    np_array.astype(np.float32).tofile(filename)
    print(f"Saved {tensor.shape} tensor to {filename} ({np_array.nbytes} bytes)")

def save_norm_params(scale, bias, filename):
    """Save layer norm parameters as float32 binary."""
    if bias is not None:
        combined = torch.stack([scale, bias])
    else:
        combined = scale.unsqueeze(0)
    combined.detach().cpu().numpy().astype(np.float32).tofile(filename)
    print(f"Saved norm params to {filename} ({combined.numel() * 4} bytes)")

def main():
    MODEL_NAME = "arnir0/Tiny-LLM"
    OUTPUT_DIR = "esp32_float32_weights"
    
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
            "use_quantization": False,  # Flag to indicate no quantization
        }, f, indent=2)
    
    print("\nExtracting float32 model weights...")
    
    # 1. Input embeddings (vocab_size, hidden_size)
    embed_tokens = model.model.embed_tokens.weight
    save_tensor_binary(embed_tokens, os.path.join(OUTPUT_DIR, "embed_tokens.bin"))
    
    # 2. Layer 0 components
    layer = model.model.layers[0]
    
    # Layer norm 1 (RMSNorm)
    norm1_scale = layer.input_layernorm.weight
    norm1_bias = getattr(layer.input_layernorm, 'bias', None)
    save_norm_params(norm1_scale, norm1_bias, os.path.join(OUTPUT_DIR, "norm1.bin"))
    
    # Self-attention projections
    # Q projection (hidden_size, hidden_size) - full projection for 2 heads
    q_proj = layer.self_attn.q_proj.weight
    save_tensor_binary(q_proj, os.path.join(OUTPUT_DIR, "q_proj.bin"))
    
    # K projection (num_kv_heads * head_dim, hidden_size) - 1 KV head
    k_proj = layer.self_attn.k_proj.weight
    save_tensor_binary(k_proj, os.path.join(OUTPUT_DIR, "k_proj.bin"))
    
    # V projection (num_kv_heads * head_dim, hidden_size) - 1 KV head
    v_proj = layer.self_attn.v_proj.weight
    save_tensor_binary(v_proj, os.path.join(OUTPUT_DIR, "v_proj.bin"))
    
    # Output projection (hidden_size, hidden_size)
    o_proj = layer.self_attn.o_proj.weight
    save_tensor_binary(o_proj, os.path.join(OUTPUT_DIR, "o_proj.bin"))
    
    # Layer norm 2 (RMSNorm)
    norm2_scale = layer.post_attention_layernorm.weight
    norm2_bias = getattr(layer.post_attention_layernorm, 'bias', None)
    save_norm_params(norm2_scale, norm2_bias, os.path.join(OUTPUT_DIR, "norm2.bin"))
    
    # MLP projections (SwiGLU)
    # Gate projection (intermediate_size, hidden_size)
    gate_proj = layer.mlp.gate_proj.weight
    save_tensor_binary(gate_proj, os.path.join(OUTPUT_DIR, "gate_proj.bin"))
    
    # Up projection (intermediate_size, hidden_size)
    up_proj = layer.mlp.up_proj.weight
    save_tensor_binary(up_proj, os.path.join(OUTPUT_DIR, "up_proj.bin"))
    
    # Down projection (hidden_size, intermediate_size)
    down_proj = layer.mlp.down_proj.weight
    save_tensor_binary(down_proj, os.path.join(OUTPUT_DIR, "down_proj.bin"))
    
    # 3. Final layer norm (RMSNorm)
    final_norm_scale = model.model.norm.weight
    final_norm_bias = getattr(model.model.norm, 'bias', None)
    save_norm_params(final_norm_scale, final_norm_bias, os.path.join(OUTPUT_DIR, "final_norm.bin"))
    
    # 4. LM head (vocab_size, hidden_size)
    lm_head = model.lm_head.weight
    save_tensor_binary(lm_head, os.path.join(OUTPUT_DIR, "lm_head.bin"))
    
    print(f"\nFloat32 weight extraction complete! Files saved to {OUTPUT_DIR}/")
    print("\nFile sizes:")
    total_size = 0
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('.bin'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, file))
            total_size += size
            print(f"  {file}: {size:,} bytes ({size/1024/1024:.2f} MB)")
    
    print(f"\nTotal float32 model size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"Original model size (estimated): {sum(p.numel() for p in model.parameters()) * 4:,} bytes ({sum(p.numel() for p in model.parameters()) * 4/1024/1024:.2f} MB)")
    
    # Memory requirements analysis
    print(f"\nMemory Requirements Analysis:")
    print(f"  Total model size: {total_size/1024/1024:.2f} MB")
    print(f"  Largest single tensor: {max(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if f.endswith('.bin'))/1024/1024:.2f} MB")
    print(f"  ESP32-CAM PSRAM: 4 MB (recommended)")
    print(f"  ESP32-CAM SRAM: 520 KB")
    
    if total_size > 4 * 1024 * 1024:
        print(f"  [WARNING] Model size ({total_size/1024/1024:.2f} MB) exceeds PSRAM (4 MB)")
        print(f"  [RECOMMENDATION] Use chunked loading for large tensors")
    
    print(f"\nStreaming Strategy:")
    print(f"  1. Embeddings: {os.path.getsize(os.path.join(OUTPUT_DIR, 'embed_tokens.bin'))/1024/1024:.2f} MB - Load per token")
    print(f"  2. LM Head: {os.path.getsize(os.path.join(OUTPUT_DIR, 'lm_head.bin'))/1024/1024:.2f} MB - Load in chunks")
    print(f"  3. MLP layers: ~{os.path.getsize(os.path.join(OUTPUT_DIR, 'gate_proj.bin'))/1024:.0f} KB each - Load sequentially")
    print(f"  4. Attention: ~{os.path.getsize(os.path.join(OUTPUT_DIR, 'q_proj.bin'))/1024:.0f} KB each - Load sequentially")

if __name__ == "__main__":
    main()

