#!/usr/bin/env python3
"""
Test script to verify the quantized Tiny-LLM model works correctly.
This script loads the quantized weights and compares inference results
with the original float32 model to ensure quantization accuracy.
"""

import torch
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class QuantizedTinyLLM:
    """Simulated quantized model for testing purposes."""
    
    def __init__(self, weights_dir="esp32_weights"):
        self.weights_dir = weights_dir
        self.config = self._load_config()
        self.scales = self._load_scales()
        self.device = torch.device("cpu")
        
        # Model dimensions
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.intermediate_size = self.config["intermediate_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.rms_norm_eps = self.config["rms_norm_eps"]
        
        print(f"Initialized quantized model with config: {self.config}")
        
    def _load_config(self):
        """Load model configuration."""
        with open(os.path.join(self.weights_dir, "model_config.json"), "r") as f:
            return json.load(f)
    
    def _load_scales(self):
        """Load quantization scales."""
        with open(os.path.join(self.weights_dir, "quantization_scales.json"), "r") as f:
            return json.load(f)
    
    def _load_quantized_tensor(self, filename, shape, scale):
        """Load and dequantize a tensor."""
        filepath = os.path.join(self.weights_dir, filename)
        quantized_data = np.fromfile(filepath, dtype=np.int8).reshape(shape)
        dequantized = torch.from_numpy(quantized_data.astype(np.float32)) * scale
        return dequantized
    
    def _load_norm_params(self, filename):
        """Load layer normalization parameters."""
        filepath = os.path.join(self.weights_dir, filename)
        data = np.fromfile(filepath, dtype=np.float32)
        if len(data) == 2:
            return torch.tensor(data[0]), torch.tensor(data[1])
        else:
            return torch.tensor(data[0]), None
    
    def rms_norm(self, x, scale, bias=None):
        """RMS normalization."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        x = x * scale
        if bias is not None:
            x = x + bias
        return x
    
    def silu(self, x):
        """SiLU activation function."""
        return x * torch.sigmoid(x)
    
    def forward(self, input_ids):
        """Forward pass through the quantized model."""
        batch_size, seq_len = input_ids.shape
        
        # Load embeddings
        embed_scale = self.scales["embed_tokens"]
        embeddings = self._load_quantized_tensor("embed_tokens.bin", 
                                               (self.vocab_size, self.hidden_size), 
                                               embed_scale)
        
        # Get input embeddings
        hidden_states = embeddings[input_ids[0]]  # (seq_len, hidden_size)
        
        # Load layer norm 1
        norm1_scale, norm1_bias = self._load_norm_params("norm1.bin")
        hidden_states = self.rms_norm(hidden_states, norm1_scale, norm1_bias)
        
        # Self-attention
        hidden_states = self._self_attention(hidden_states)
        
        # Load layer norm 2
        norm2_scale, norm2_bias = self._load_norm_params("norm2.bin")
        hidden_states = self.rms_norm(hidden_states, norm2_scale, norm2_bias)
        
        # MLP
        hidden_states = self._mlp_forward(hidden_states)
        
        # Final layer norm
        final_norm_scale, final_norm_bias = self._load_norm_params("final_norm.bin")
        hidden_states = self.rms_norm(hidden_states, final_norm_scale, final_norm_bias)
        
        # LM head
        lm_head_scale = self.scales["lm_head"]
        lm_head = self._load_quantized_tensor("lm_head.bin",
                                            (self.vocab_size, self.hidden_size),
                                            lm_head_scale)
        
        # Compute logits
        logits = torch.matmul(hidden_states, lm_head.T)  # (seq_len, vocab_size)
        
        return logits
    
    def _self_attention(self, hidden_states):
        """Self-attention computation."""
        seq_len = hidden_states.size(0)
        
        # Load attention projections
        q_scale = self.scales["q_proj"]
        k_scale = self.scales["k_proj"]
        v_scale = self.scales["v_proj"]
        o_scale = self.scales["o_proj"]
        
        q_proj = self._load_quantized_tensor("q_proj.bin", 
                                           (self.hidden_size, self.hidden_size), q_scale)
        k_proj = self._load_quantized_tensor("k_proj.bin",
                                           (self.num_kv_heads * self.head_dim, self.hidden_size), k_scale)
        v_proj = self._load_quantized_tensor("v_proj.bin",
                                           (self.num_kv_heads * self.head_dim, self.hidden_size), v_scale)
        o_proj = self._load_quantized_tensor("o_proj.bin",
                                           (self.hidden_size, self.hidden_size), o_scale)
        
        # Compute Q, K, V
        q = torch.matmul(hidden_states, q_proj.T)  # (seq_len, hidden_size)
        k = torch.matmul(hidden_states, k_proj.T)  # (seq_len, num_kv_heads * head_dim)
        v = torch.matmul(hidden_states, v_proj.T)  # (seq_len, num_kv_heads * head_dim)
        
        # Reshape for attention
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim)
        
        # Expand K, V for grouped query attention
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, v)
        attention_output = attention_output.view(seq_len, self.hidden_size)
        
        # Output projection
        output = torch.matmul(attention_output, o_proj.T)
        
        # Residual connection
        return hidden_states + output
    
    def _mlp_forward(self, hidden_states):
        """MLP forward pass with SwiGLU."""
        # Load MLP projections
        gate_scale = self.scales["gate_proj"]
        up_scale = self.scales["up_proj"]
        down_scale = self.scales["down_proj"]
        
        gate_proj = self._load_quantized_tensor("gate_proj.bin",
                                              (self.intermediate_size, self.hidden_size), gate_scale)
        up_proj = self._load_quantized_tensor("up_proj.bin",
                                            (self.intermediate_size, self.hidden_size), up_scale)
        down_proj = self._load_quantized_tensor("down_proj.bin",
                                              (self.hidden_size, self.intermediate_size), down_scale)
        
        # SwiGLU computation
        gate = torch.matmul(hidden_states, gate_proj.T)
        up = torch.matmul(hidden_states, up_proj.T)
        
        # Apply SiLU to gate and multiply with up
        gate_silu = self.silu(gate)
        intermediate = gate_silu * up
        
        # Down projection
        down = torch.matmul(intermediate, down_proj.T)
        
        # Residual connection
        return hidden_states + down

def test_quantized_model():
    """Test the quantized model against the original model."""
    print("=" * 60)
    print("Testing Quantized Tiny-LLM Model")
    print("=" * 60)
    
    # Load original model
    print("\n1. Loading original Tiny-LLM model...")
    MODEL_NAME = "arnir0/Tiny-LLM"
    original_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load quantized model
    print("2. Loading quantized model...")
    quantized_model = QuantizedTinyLLM("esp32_weights")
    
    # Test with sample input
    test_prompts = [
        "Hello, how are you?",
        "The quick brown fox",
        "In a hole in the ground",
        "Once upon a time",
        "The future of AI is"
    ]
    
    print(f"\n3. Testing with {len(test_prompts)} sample prompts...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1}: '{prompt}' ---")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"Input tokens: {input_ids[0].tolist()}")
        print(f"Input shape: {input_ids.shape}")
        
        # Get original model output
        with torch.no_grad():
            original_outputs = original_model(input_ids)
            original_logits = original_outputs.logits[0, -1, :]  # Last token logits
            original_probs = torch.softmax(original_logits, dim=-1)
            original_top5 = torch.topk(original_probs, 5)
        
        # Get quantized model output
        with torch.no_grad():
            quantized_logits = quantized_model.forward(input_ids)[-1, :]  # Last token logits
            quantized_probs = torch.softmax(quantized_logits, dim=-1)
            quantized_top5 = torch.topk(quantized_probs, 5)
        
        # Compare results
        print(f"\nOriginal model top-5 predictions:")
        for j, (prob, idx) in enumerate(zip(original_top5.values, original_top5.indices)):
            token = tokenizer.decode([idx])
            print(f"  {j+1}. '{token}' (prob: {prob:.4f}, token_id: {idx})")
        
        print(f"\nQuantized model top-5 predictions:")
        for j, (prob, idx) in enumerate(zip(quantized_top5.values, quantized_top5.indices)):
            token = tokenizer.decode([idx])
            print(f"  {j+1}. '{token}' (prob: {prob:.4f}, token_id: {idx})")
        
        # Calculate similarity metrics
        logits_diff = torch.abs(original_logits - quantized_logits)
        max_diff = torch.max(logits_diff).item()
        mean_diff = torch.mean(logits_diff).item()
        
        # KL divergence
        kl_div = torch.sum(original_probs * torch.log(original_probs / (quantized_probs + 1e-8))).item()
        
        print(f"\nQuantization accuracy metrics:")
        print(f"  Max logit difference: {max_diff:.4f}")
        print(f"  Mean logit difference: {mean_diff:.4f}")
        print(f"  KL divergence: {kl_div:.6f}")
        
        # Check if top predictions match
        original_top_token = original_top5.indices[0].item()
        quantized_top_token = quantized_top5.indices[0].item()
        top_match = original_top_token == quantized_top_token
        
        print(f"  Top token match: {'[OK]' if top_match else '[DIFF]'}")
        
        if top_match:
            print("  [OK] Quantization successful for this prompt!")
        else:
            print("  [WARN] Top token differs - check quantization quality")
    
    # Test memory usage and performance
    print(f"\n4. Performance and memory analysis...")
    
    # Check file sizes
    total_size = 0
    print(f"\nQuantized model file sizes:")
    for file in os.listdir("esp32_weights"):
        if file.endswith('.bin'):
            size = os.path.getsize(os.path.join("esp32_weights", file))
            total_size += size
            print(f"  {file}: {size:,} bytes ({size/1024:.1f} KB)")
    
    original_size = sum(p.numel() for p in original_model.parameters()) * 4  # 4 bytes per float32
    compression_ratio = original_size / total_size
    
    print(f"\nSize comparison:")
    print(f"  Original model: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
    print(f"  Quantized model: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    # Test generation
    print(f"\n5. Testing text generation...")
    
    test_prompt = "The future of artificial intelligence is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    print(f"Prompt: '{test_prompt}'")
    
    # Generate with original model
    with torch.no_grad():
        original_outputs = original_model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        original_generated = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    
    print(f"Original model output: '{original_generated}'")
    
    # Note: For quantized model generation, we'd need to implement the full generation loop
    # For now, just show the next token prediction
    with torch.no_grad():
        quantized_logits = quantized_model.forward(inputs["input_ids"])
        next_token_id = torch.argmax(quantized_logits[-1, :]).item()
        next_token = tokenizer.decode([next_token_id])
    
    print(f"Quantized model next token: '{next_token}' (id: {next_token_id})")
    
    print(f"\n" + "=" * 60)
    print("Quantization test completed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        test_quantized_model()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()