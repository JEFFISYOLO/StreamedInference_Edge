#include "fixed_point_math.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static const char* TAG = "fixed_point_math";

// Fixed-point arithmetic operations
int32_t fixed_multiply(int32_t a, int32_t b) {
    int64_t result = (int64_t)a * b;
    return (int32_t)(result >> FIXED_POINT_SHIFT);
}

int32_t fixed_add(int32_t a, int32_t b) {
    return a + b;
}

int32_t fixed_sqrt(int32_t value) {
    if (value <= 0) return 0;
    float float_val = fixed_to_float(value);
    float sqrt_val = sqrtf(float_val);
    return float_to_fixed(sqrt_val);
}

int32_t fixed_exp(int32_t value) {
    float float_val = fixed_to_float(value);
    float exp_val = expf(float_val);
    return float_to_fixed(exp_val);
}

int32_t fixed_reciprocal_sqrt(int32_t value) {
    if (value <= 0) return 0;
    float float_val = fixed_to_float(value);
    float rsqrt_val = 1.0f / sqrtf(float_val);
    return float_to_fixed(rsqrt_val);
}

// Vector operations
void vector_add_fixed(int32_t* result, const int32_t* a, const int32_t* b, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_scale_fixed(int32_t* result, const int32_t* input, int32_t scale, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = fixed_multiply(input[i], scale);
    }
}

void vector_add_scalar_fixed(int32_t* result, const int32_t* input, int32_t scalar, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = input[i] + scalar;
    }
}

// Matrix-vector multiplication (quantized input, fixed-point output)
void matvec_quant_to_fixed(int32_t* output, const int8_t* matrix, const int32_t* input,
                          int rows, int cols, float quant_scale) {
    int32_t scale_fixed = float_to_fixed(quant_scale);
    
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            int32_t matrix_val = quant_to_fixed(matrix[i * cols + j], 1.0f);
            sum += fixed_multiply(matrix_val, input[j]);
        }
        output[i] = fixed_multiply(sum, scale_fixed);
    }
}

// RMS normalization
void rms_norm_fixed(int32_t* output, const int32_t* input, const float* scale, 
                   const float* bias, int length, float eps) {
    // Calculate sum of squares
    int64_t sum_squares = 0;
    for (int i = 0; i < length; i++) {
        int64_t val = input[i];
        sum_squares += val * val;
    }
    
    // Calculate RMS
    int32_t mean_squares = (int32_t)(sum_squares / length);
    int32_t rms = fixed_sqrt(mean_squares);
    
    // Add epsilon and calculate reciprocal
    int32_t eps_fixed = float_to_fixed(eps);
    int32_t rms_with_eps = rms + eps_fixed;
    int32_t rms_reciprocal = fixed_reciprocal_sqrt(rms_with_eps);
    
    // Normalize and apply scale and bias
    for (int i = 0; i < length; i++) {
        int32_t normalized = fixed_multiply(input[i], rms_reciprocal);
        int32_t scaled = fixed_multiply(normalized, float_to_fixed(scale[i]));
        output[i] = scaled + float_to_fixed(bias[i]);
    }
}

// SiLU activation function
void silu_fixed(int32_t* output, const int32_t* input, int length) {
    for (int i = 0; i < length; i++) {
        // SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        int32_t neg_x = -input[i];
        int32_t exp_neg_x = fixed_exp(neg_x);
        int32_t one_plus_exp = float_to_fixed(1.0f) + exp_neg_x;
        int32_t sigmoid = float_to_fixed(1.0f) / fixed_to_float(one_plus_exp);
        output[i] = fixed_multiply(input[i], sigmoid);
    }
}

// Softmax (approximated for efficiency)
void softmax_fixed(int32_t* output, const int32_t* input, int length) {
    // Find maximum for numerical stability
    int32_t max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Calculate exponentials and sum
    int32_t sum = 0;
    int32_t* exp_vals = malloc(length * sizeof(int32_t));
    for (int i = 0; i < length; i++) {
        exp_vals[i] = fixed_exp(input[i] - max_val);
        sum += exp_vals[i];
    }
    
    // Normalize
    for (int i = 0; i < length; i++) {
        output[i] = fixed_multiply(exp_vals[i], float_to_fixed(1.0f / fixed_to_float(sum)));
    }
    
    free(exp_vals);
}

// Attention score computation
void compute_attention_scores_fixed(int32_t* scores, const int32_t* query, const int32_t* key,
                                   int seq_len, int head_dim, int32_t scale_factor) {
    for (int i = 0; i < seq_len; i++) {
        int32_t score = 0;
        for (int j = 0; j < head_dim; j++) {
            score += fixed_multiply(query[j], key[i * head_dim + j]);
        }
        scores[i] = fixed_multiply(score, scale_factor);
    }
}

// Top-k sampling
uint16_t sample_top_k(int32_t* logits, int vocab_size, int top_k) {
    // Simple implementation - find top_k indices and sample randomly
    // In practice, you'd want a more efficient implementation
    int32_t max_logit = logits[0];
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
uint16_t sample_top_p(int32_t* logits, int vocab_size, float top_p) {
    // Simplified implementation - in practice, you'd sort and cumsum
    int32_t max_logit = logits[0];
    uint16_t max_idx = 0;
    
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

