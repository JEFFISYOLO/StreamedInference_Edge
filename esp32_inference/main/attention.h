#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdint.h>
#include "tinyllm_inference.h"

// Attention computation structure
typedef struct {
    float query[HIDDEN_SIZE];
    float key[NUM_KV_HEADS * HEAD_DIM];
    float value[NUM_KV_HEADS * HEAD_DIM];
    float attention_scores[MAX_SEQ_LEN];
    float attention_output[HIDDEN_SIZE];
} attention_state_t;

// Function prototypes
esp_err_t compute_self_attention(float* hidden_state, int position, 
                                hidden_t* kv_cache_k, hidden_t* kv_cache_v,
                                attention_state_t* attn_state);

esp_err_t load_attention_weights_and_compute(float* hidden_state, int position,
                                            hidden_t* kv_cache_k, hidden_t* kv_cache_v,
                                            attention_state_t* attn_state);

// Individual attention components
esp_err_t compute_query_projection(float* query, const float* hidden_state);
esp_err_t compute_key_projection(float* key, const float* hidden_state);
esp_err_t compute_value_projection(float* value, const float* hidden_state);
esp_err_t compute_attention_scores_and_apply(float* scores, const float* query,
                                            hidden_t* kv_cache_k, int position);
esp_err_t apply_attention_weights(float* attention_output, const float* scores,
                                 hidden_t* kv_cache_v, const float* value, int position);
esp_err_t compute_output_projection(float* output, const float* attention_output);

// KV cache management
void update_kv_cache(hidden_t* kv_cache_k, hidden_t* kv_cache_v,
                    const float* key, const float* value, int position);
void clear_kv_cache(hidden_t* kv_cache_k, hidden_t* kv_cache_v);

#endif // ATTENTION_H
