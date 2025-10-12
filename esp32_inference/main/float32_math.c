#include "float32_math.h"
#include "esp_log.h"
#include <string.h>
#include <stdlib.h>

static const char* TAG = "float32_math";

// Vector operations
void vector_add_float(float* result, const float* a, const float* b, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_scale_float(float* result, const float* input, float scale, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = input[i] * scale;
    }
}

void vector_add_scalar_float(float* result, const float* input, float scalar, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = input[i] + scalar;
    }
}

// Matrix-vector multiplication (float32)
void matvec_float(float* output, const float* matrix, const float* input,
                 int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * input[j];
        }
        output[i] = sum;
    }
}

// RMS normalization
void rms_norm_float(float* output, const float* input, const float* scale, 
                   const float* bias, int length, float eps) {
    // Calculate sum of squares
    float sum_squares = 0.0f;
    for (int i = 0; i < length; i++) {
        sum_squares += input[i] * input[i];
    }
    
    // Calculate RMS
    float mean_squares = sum_squares / length;
    float rms = sqrtf(mean_squares + eps);
    
    // Normalize and apply scale and bias
    for (int i = 0; i < length; i++) {
        float normalized = input[i] / rms;
        float scaled = normalized * scale[i];
        output[i] = scaled + (bias ? bias[i] : 0.0f);
    }
}

// SiLU activation function
void silu_float(float* output, const float* input, int length) {
    for (int i = 0; i < length; i++) {
        // SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        float sigmoid_val = 1.0f / (1.0f + expf(-input[i]));
        output[i] = input[i] * sigmoid_val;
    }
}

// Softmax (float32)
void softmax_float(float* output, const float* input, int length) {
    // Find maximum for numerical stability
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Calculate exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// Attention score computation
void compute_attention_scores_float(float* scores, const float* query, const float* key,
                                   int seq_len, int head_dim, float scale_factor) {
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            score += query[j] * key[i * head_dim + j];
        }
        scores[i] = score * scale_factor;
    }
}

// Top-k sampling
uint16_t sample_top_k_float(float* logits, int vocab_size, int top_k) {
    // Simple implementation - find top_k indices and sample randomly
    // In practice, you'd want a more efficient implementation
    float max_logit = logits[0];
    uint16_t max_idx = 0;
    
    for (int i = 1; i < vocab_size && i < top_k; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// Top-p (nucleus) sampling
uint16_t sample_top_p_float(float* logits, int vocab_size, float top_p) {
    // Simplified implementation - in practice, you'd sort and cumsum
    float max_logit = logits[0];
    uint16_t max_idx = 0;
    
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// Utility functions
float sigmoid_float(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float tanh_float(float x) {
    return tanhf(x);
}

float exp_float(float x) {
    return expf(x);
}

