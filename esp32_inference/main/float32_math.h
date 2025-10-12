#ifndef FLOAT32_MATH_H
#define FLOAT32_MATH_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "tinyllm_inference.h"

// Vector operations
void vector_add_float(float* result, const float* a, const float* b, int length);
void vector_scale_float(float* result, const float* input, float scale, int length);
void vector_add_scalar_float(float* result, const float* input, float scalar, int length);

// Matrix-vector multiplication (float32)
void matvec_float(float* output, const float* matrix, const float* input,
                 int rows, int cols);

// RMS normalization
void rms_norm_float(float* output, const float* input, const float* scale, 
                   const float* bias, int length, float eps);

// Activation functions
void silu_float(float* output, const float* input, int length);
void softmax_float(float* output, const float* input, int length);

// Attention computation
void compute_attention_scores_float(float* scores, const float* query, const float* key,
                                   int seq_len, int head_dim, float scale_factor);

// Sampling functions
uint16_t sample_top_k_float(float* logits, int vocab_size, int top_k);
uint16_t sample_top_p_float(float* logits, int vocab_size, float top_p);

// Utility functions
float sigmoid_float(float x);
float tanh_float(float x);
float exp_float(float x);

#endif // FLOAT32_MATH_H

