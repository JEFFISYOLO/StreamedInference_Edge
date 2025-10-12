#include "attention.h"
#include "weight_loader.h"
#include "float32_math.h"
#include "esp_log.h"
#include <string.h>
#include <stdlib.h>
#include <esp_heap_caps.h>

static const char* TAG = "attention";

// Temporary buffer for streaming computation (heap-allocated)
static float* weight_buffer = NULL;

esp_err_t compute_self_attention(float* hidden_state, int position, 
                                hidden_t* kv_cache_k, hidden_t* kv_cache_v,
                                attention_state_t* attn_state) {
    ESP_LOGD(TAG, "Computing self-attention for position %d", position);
    
    // 1. Compute query projection
    ESP_ERROR_CHECK(compute_query_projection(attn_state->query, hidden_state));
    
    // 2. Compute key projection
    ESP_ERROR_CHECK(compute_key_projection(attn_state->key, hidden_state));
    
    // 3. Compute value projection
    ESP_ERROR_CHECK(compute_value_projection(attn_state->value, hidden_state));
    
    // 4. Update KV cache
    update_kv_cache(kv_cache_k, kv_cache_v, attn_state->key, attn_state->value, position);
    
    // 5. Compute attention scores
    ESP_ERROR_CHECK(compute_attention_scores_and_apply(attn_state->attention_scores, 
                                                      attn_state->query, kv_cache_k, position));
    
    // 6. Apply attention weights
    ESP_ERROR_CHECK(apply_attention_weights(attn_state->attention_output, 
                                           attn_state->attention_scores, 
                                           kv_cache_v, attn_state->value, position));
    
    // 7. Compute output projection
    ESP_ERROR_CHECK(compute_output_projection(hidden_state, attn_state->attention_output));
    
    return ESP_OK;
}

esp_err_t compute_query_projection(float* query, const float* hidden_state) {
    // Ensure weight buffer is allocated
    if (!weight_buffer) {
        weight_buffer = heap_caps_malloc(MAX_CHUNK_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!weight_buffer) {
            ESP_LOGE(TAG, "Failed to allocate weight_buffer");
            return ESP_ERR_NO_MEM;
        }
    }

    // Load Q projection weights
    esp_err_t ret = load_projection_matrix(Q_PROJ_FILE, weight_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load Q projection weights");
        return ret;
    }
    
    // Matrix-vector multiplication: query = hidden_state @ q_proj.T
    matvec_float(query, weight_buffer, hidden_state, HIDDEN_SIZE, HIDDEN_SIZE);
    
    return ESP_OK;
}

esp_err_t compute_key_projection(float* key, const float* hidden_state) {
    // Ensure weight buffer is allocated
    if (!weight_buffer) {
        weight_buffer = heap_caps_malloc(MAX_CHUNK_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!weight_buffer) {
            ESP_LOGE(TAG, "Failed to allocate weight_buffer");
            return ESP_ERR_NO_MEM;
        }
    }

    // Load K projection weights
    esp_err_t ret = load_projection_matrix(K_PROJ_FILE, weight_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load K projection weights");
        return ret;
    }
    
    // Matrix-vector multiplication: key = hidden_state @ k_proj.T
    matvec_float(key, weight_buffer, hidden_state, NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
    
    return ESP_OK;
}

esp_err_t compute_value_projection(float* value, const float* hidden_state) {
    // Ensure weight buffer is allocated
    if (!weight_buffer) {
        weight_buffer = heap_caps_malloc(MAX_CHUNK_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!weight_buffer) {
            ESP_LOGE(TAG, "Failed to allocate weight_buffer");
            return ESP_ERR_NO_MEM;
        }
    }

    // Load V projection weights
    esp_err_t ret = load_projection_matrix(V_PROJ_FILE, weight_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load V projection weights");
        return ret;
    }
    
    // Matrix-vector multiplication: value = hidden_state @ v_proj.T
    matvec_float(value, weight_buffer, hidden_state, NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
    
    return ESP_OK;
}

esp_err_t compute_attention_scores_and_apply(float* scores, const float* query,
                                           hidden_t* kv_cache_k, int position) {
    // Compute attention scores: scores[i] = query @ kv_cache_k[i] / sqrt(HEAD_DIM)
    float scale_factor = 1.0f / sqrtf(HEAD_DIM);
    
    for (int i = 0; i <= position; i++) {
        float score = 0.0f;
        for (int j = 0; j < HEAD_DIM; j++) {
            score += query[j] * kv_cache_k[i * HEAD_DIM + j];
        }
        scores[i] = score * scale_factor;
    }
    
    // Apply softmax
    softmax_float(scores, scores, position + 1);
    
    return ESP_OK;
}

esp_err_t apply_attention_weights(float* attention_output, const float* scores,
                                hidden_t* kv_cache_v, const float* current_value, int position) {
    // Initialize output
    for (int j = 0; j < HEAD_DIM; j++) {
        attention_output[j] = 0.0f;
    }
    
    // Apply attention weights to cached values
    for (int i = 0; i <= position; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            attention_output[j] += scores[i] * kv_cache_v[i * HEAD_DIM + j];
        }
    }
    
    // Apply attention weights to current value
    for (int j = 0; j < HEAD_DIM; j++) {
        attention_output[j] += scores[position] * current_value[j];
    }
    
    return ESP_OK;
}

esp_err_t compute_output_projection(float* output, const float* attention_output) {
    // Ensure weight buffer is allocated
    if (!weight_buffer) {
        weight_buffer = heap_caps_malloc(MAX_CHUNK_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!weight_buffer) {
            ESP_LOGE(TAG, "Failed to allocate weight_buffer");
            return ESP_ERR_NO_MEM;
        }
    }

    // Load O projection weights
    esp_err_t ret = load_projection_matrix(O_PROJ_FILE, weight_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load O projection weights");
        return ret;
    }
    
    // Matrix-vector multiplication: output = attention_output @ o_proj.T
    matvec_float(output, weight_buffer, attention_output, HIDDEN_SIZE, HIDDEN_SIZE);
    
    return ESP_OK;
}

void update_kv_cache(hidden_t* kv_cache_k, hidden_t* kv_cache_v,
                    const float* key, const float* value, int position) {
    // Store key and value in cache at current position
    for (int i = 0; i < HEAD_DIM; i++) {
        kv_cache_k[position * HEAD_DIM + i] = key[i];
        kv_cache_v[position * HEAD_DIM + i] = value[i];
    }
}

void clear_kv_cache(hidden_t* kv_cache_k, hidden_t* kv_cache_v) {
    // Clear the entire KV cache
    memset(kv_cache_k, 0, KV_CACHE_SIZE * sizeof(hidden_t));
    memset(kv_cache_v, 0, KV_CACHE_SIZE * sizeof(hidden_t));
}