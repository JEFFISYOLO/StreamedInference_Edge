#ifndef TINYLLM_INFERENCE_H
#define TINYLLM_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

// Model configuration constants
#define VOCAB_SIZE 32000
#define HIDDEN_SIZE 192
#define NUM_LAYERS 1
#define NUM_HEADS 2
#define NUM_KV_HEADS 1
#define HEAD_DIM (HIDDEN_SIZE / NUM_HEADS)
#define INTERMEDIATE_SIZE 1024
#define MAX_SEQ_LEN 1024
#define RMS_NORM_EPS 1e-5f

// Memory management for float32
#define MAX_CHUNK_SIZE (512 * 1024)  // 512KB max chunk size for float32
#define KV_CACHE_SIZE (MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM)

// Data types for float32 inference
typedef float weight_t;
typedef float hidden_t;
typedef float accumulator_t;

// Model state structure
typedef struct {
    hidden_t hidden_state[HIDDEN_SIZE];
    // Move KV caches to heap to avoid large .bss growth
    hidden_t* kv_cache_k;
    hidden_t* kv_cache_v;
    int32_t position;
    bool cache_valid;
} model_state_t;

// Inference configuration
typedef struct {
    float temperature;
    int top_k;
    float top_p;
    int max_new_tokens;
    bool use_kv_cache;
} inference_config_t;

// Function prototypes
esp_err_t tinyllm_init(void);
esp_err_t tinyllm_inference(const uint16_t* prompt_tokens, int prompt_len,
                           uint16_t* output_tokens, int max_output_len,
                           const inference_config_t* config);
void tinyllm_deinit(void);

// Utility functions
esp_err_t load_model_weights_from_sd(void);
uint16_t sample_token(const float* logits, const inference_config_t* config);
float sigmoid(float x);
float softmax_sum(const float* logits, int length);

#endif // TINYLLM_INFERENCE_H
