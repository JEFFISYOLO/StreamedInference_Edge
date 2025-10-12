#include "lm_head.h"
#include "weight_loader.h"
#include "float32_math.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>
#include <stdlib.h>
#include <esp_heap_caps.h>

static const char* TAG = "lm_head";

// Temporary buffer for weight chunks (heap-allocated)
static float* weight_buffer = NULL; // allocated on first use

esp_err_t compute_lm_head_chunked(float* hidden_state, lm_head_state_t* lm_head_state) {
    ESP_LOGD(TAG, "Computing LM head chunked");
    
    // Initialize logits to zero
    // Ensure lm_head_state->logits is allocated by caller; if NULL, allocate temporarily
    if (!lm_head_state->logits) {
        lm_head_state->logits = heap_caps_malloc(VOCAB_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!lm_head_state->logits) {
            ESP_LOGE(TAG, "Failed to allocate lm_head_state logits");
            return ESP_ERR_NO_MEM;
        }
    }
    memset(lm_head_state->logits, 0, VOCAB_SIZE * sizeof(float));
    
    int chunk_size = 256; // Process 256 tokens at a time
    int num_chunks = (VOCAB_SIZE + chunk_size - 1) / chunk_size;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_idx = chunk * chunk_size;
        int end_idx = (start_idx + chunk_size < VOCAB_SIZE) ? start_idx + chunk_size : VOCAB_SIZE;
        int actual_chunk_size = end_idx - start_idx;
        
        // Ensure temporary buffer for chunk exists
        if (!lm_head_state->temp_logits) {
            // temp_logits is small (chunk_size up to 256), allocate on heap to be safe
            lm_head_state->temp_logits = heap_caps_malloc(256 * sizeof(float), MALLOC_CAP_DEFAULT);
            if (!lm_head_state->temp_logits) {
                ESP_LOGE(TAG, "Failed to allocate temp_logits");
                return ESP_ERR_NO_MEM;
            }
        }
        // Compute chunk
        ESP_ERROR_CHECK(compute_lm_head_chunk(lm_head_state->temp_logits, hidden_state,
                                            actual_chunk_size, chunk));
        
    // Merge into final logits
    ESP_ERROR_CHECK(merge_lm_head_chunks(&lm_head_state->logits[start_idx], 
                       lm_head_state->temp_logits, actual_chunk_size, 1));
    }
    
    return ESP_OK;
}

esp_err_t compute_lm_head_chunk(float* chunk_logits, const float* hidden_state,
                               int chunk_size, int chunk_index) {
    // Load weight chunk
    // Ensure weight buffer allocation
    if (!weight_buffer) {
        weight_buffer = heap_caps_malloc(MAX_CHUNK_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!weight_buffer) {
            ESP_LOGE(TAG, "Failed to allocate weight_buffer");
            return ESP_ERR_NO_MEM;
        }
    }

    int64_t t0 = esp_timer_get_time();
    esp_err_t ret = load_lm_head_chunk(weight_buffer, chunk_size, chunk_index);
    int64_t t_after_io = esp_timer_get_time();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load LM head chunk %d", chunk_index);
        return ret;
    }

    // Matrix-vector multiplication: logits = hidden_state @ lm_head.T
    int64_t t_compute_start = esp_timer_get_time();
    matvec_float(chunk_logits, weight_buffer, hidden_state, chunk_size, HIDDEN_SIZE);
    int64_t t_compute_end = esp_timer_get_time();

    ESP_LOGD(TAG, "LM head chunk %d: IO=%lld us, compute=%lld us", chunk_index,
             (long long)(t_after_io - t0), (long long)(t_compute_end - t_compute_start));

    return ESP_OK;
}

esp_err_t merge_lm_head_chunks(float* final_logits, const float* chunk_logits,
                              int chunk_size, int num_chunks) {
    // Simple copy for single chunk (could be optimized for multiple chunks)
    for (int i = 0; i < chunk_size; i++) {
        final_logits[i] = chunk_logits[i];
    }
    
    return ESP_OK;
}

uint16_t sample_token(const float* logits, const inference_config_t* config) {
    // Apply temperature scaling
    float temp_scale = 1.0f / config->temperature;
    float* scaled_logits = heap_caps_malloc(VOCAB_SIZE * sizeof(float), MALLOC_CAP_DEFAULT);
    if (!scaled_logits) {
        ESP_LOGE(TAG, "Failed to allocate scaled_logits for sampling");
        return 0;
    }

    for (int i = 0; i < VOCAB_SIZE; i++) {
        scaled_logits[i] = logits[i] * temp_scale;
    }
    
    // Apply sampling strategy
    uint16_t sampled_token;
    if (config->top_k > 0) {
        sampled_token = sample_top_k_tokens(scaled_logits, VOCAB_SIZE, config->top_k);
    } else if (config->top_p > 0.0f) {
        sampled_token = sample_top_p_tokens(scaled_logits, VOCAB_SIZE, config->top_p);
    } else {
        // Greedy sampling (argmax)
        sampled_token = 0;
        float max_logit = scaled_logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (scaled_logits[i] > max_logit) {
                max_logit = scaled_logits[i];
                sampled_token = i;
            }
        }
    }
    
    heap_caps_free(scaled_logits);
    return sampled_token;
}

uint16_t sample_top_k_tokens(float* logits, int vocab_size, int top_k) {
    // Simple top-k sampling (could be optimized)
    uint16_t* indices = malloc(vocab_size * sizeof(uint16_t));
    if (!indices) {
        ESP_LOGE(TAG, "Failed to allocate memory for top-k sampling");
        return 0;
    }
    
    // Create indices array
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    // Sort by logit values (simple bubble sort for small vocab)
    for (int i = 0; i < top_k && i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[indices[j]] > logits[indices[i]]) {
                uint16_t temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Random selection from top-k
    uint16_t selected = indices[rand() % top_k];
    free(indices);
    
    return selected;
}

uint16_t sample_top_p_tokens(float* logits, int vocab_size, float top_p) {
    // Simple nucleus sampling (could be optimized)
    
    // Apply softmax to get probabilities
    softmax_float(logits, logits, vocab_size);
    
    // Sort by probability (descending)
    uint16_t* indices = malloc(vocab_size * sizeof(uint16_t));
    if (!indices) {
        ESP_LOGE(TAG, "Failed to allocate memory for top-p sampling");
        return 0;
    }
    
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    // Simple sort by probability
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[indices[j]] > logits[indices[i]]) {
                uint16_t temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Find cutoff point
    float cumulative_prob = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumulative_prob += logits[indices[i]];
        if (cumulative_prob >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    
    // Random selection from top-p tokens
    uint16_t selected = indices[rand() % cutoff];
    free(indices);
    
    return selected;
}