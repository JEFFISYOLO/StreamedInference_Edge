#ifndef FIXED_POINT_MATH_H
#define FIXED_POINT_MATH_H

#include <stdint.h>
#include <stdbool.h>
#include "tinyllm_inference.h"

// Fixed-point math utilities
#define FIXED_POINT_SHIFT 16
#define FIXED_POINT_SCALE (1 << FIXED_POINT_SHIFT)
#define FIXED_POINT_MAX ((1 << (31 - FIXED_POINT_SHIFT)) - 1)
#define FIXED_POINT_MIN (-(1 << (31 - FIXED_POINT_SHIFT)))

// Conversion functions
static inline int32_t float_to_fixed(float value) {
    return (int32_t)(value * FIXED_POINT_SCALE);
}

static inline float fixed_to_float(int32_t value) {
    return (float)value / FIXED_POINT_SCALE;
}

static inline int32_t quant_to_fixed(quant_t value, float scale) {
    return (int32_t)(value * scale * FIXED_POINT_SCALE);
}

// Fixed-point arithmetic operations
int32_t fixed_multiply(int32_t a, int32_t b);
int32_t fixed_add(int32_t a, int32_t b);
int32_t fixed_sqrt(int32_t value);
int32_t fixed_exp(int32_t value);
int32_t fixed_reciprocal_sqrt(int32_t value);

// Vector operations
void vector_add_fixed(int32_t* result, const int32_t* a, const int32_t* b, int length);
void vector_scale_fixed(int32_t* result, const int32_t* input, int32_t scale, int length);
void vector_add_scalar_fixed(int32_t* result, const int32_t* input, int32_t scalar, int length);

// Matrix-vector multiplication (quantized input, fixed-point output)
void matvec_quant_to_fixed(int32_t* output, const int8_t* matrix, const int32_t* input,
                          int rows, int cols, float quant_scale);

// RMS normalization
void rms_norm_fixed(int32_t* output, const int32_t* input, const float* scale, 
                   const float* bias, int length, float eps);

// Activation functions
void silu_fixed(int32_t* output, const int32_t* input, int length);
void softmax_fixed(int32_t* output, const int32_t* input, int length);

// Attention computation
void compute_attention_scores_fixed(int32_t* scores, const int32_t* query, const int32_t* key,
                                   int seq_len, int head_dim, int32_t scale_factor);

// Sampling functions
uint16_t sample_top_k(int32_t* logits, int vocab_size, int top_k);
uint16_t sample_top_p(int32_t* logits, int vocab_size, float top_p);

#endif // FIXED_POINT_MATH_H

