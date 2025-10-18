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
    
    // Compute an automatic token-based chunk_size so that chunk_size * HIDDEN_SIZE * sizeof(float)
    // fits within available PSRAM up to MAX_CHUNK_SIZE.
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    // Keep a safety margin to avoid exhausting PSRAM for other allocations
    size_t usable = psram_free / 2;
    if (usable > (size_t)MAX_CHUNK_SIZE) usable = MAX_CHUNK_SIZE;
    if (usable < (size_t)(HIDDEN_SIZE * sizeof(float))) usable = HIDDEN_SIZE * sizeof(float);

    int max_chunk_floats = usable / sizeof(float);
    int max_tokens_per_chunk = max_chunk_floats / HIDDEN_SIZE;
    if (max_tokens_per_chunk < 1) max_tokens_per_chunk = 1;

    int chunk_size = (max_tokens_per_chunk < VOCAB_SIZE) ? max_tokens_per_chunk : VOCAB_SIZE;
    int num_chunks = (VOCAB_SIZE + chunk_size - 1) / chunk_size;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_idx = chunk * chunk_size;
        int end_idx = (start_idx + chunk_size < VOCAB_SIZE) ? start_idx + chunk_size : VOCAB_SIZE;
        int actual_chunk_size = end_idx - start_idx;
        
        // Ensure temporary buffer for chunk exists
        if (!lm_head_state->temp_logits) {
            // Allocate temp_logits sized to the chosen chunk size
            lm_head_state->temp_logits = heap_caps_malloc(chunk_size * sizeof(float), MALLOC_CAP_DEFAULT);
            if (!lm_head_state->temp_logits) {
                ESP_LOGE(TAG, "Failed to allocate temp_logits for chunk_size=%d", chunk_size);
                return ESP_ERR_NO_MEM;
            }
        }
        // Compute chunk
    // compute_lm_head_chunk expects chunk_size in tokens (number of vocabulary columns)
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
        // Allocate a buffer large enough for the maximum allowed chunk (in floats)
        size_t bytes = MAX_CHUNK_SIZE;
        weight_buffer = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!weight_buffer) {
            ESP_LOGE(TAG, "Failed to allocate weight_buffer of %zu bytes", bytes);
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

// Reusable buffers for sampling to avoid repeated allocations.
static float* s_sampling_buffer = NULL; // temperaturescaled / probabilities
static float* s_softmax_buffer = NULL; // for softmax outputs when needed
static uint16_t* s_indices = NULL; // indices array for sorting/selection

// Ensure sampling buffers are allocated (in PSRAM) once
static bool ensure_sampling_buffers(void) {
    if (s_sampling_buffer && s_indices && s_softmax_buffer) return true;
    // Allocate in PSRAM where possible
    s_sampling_buffer = heap_caps_malloc(VOCAB_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s_softmax_buffer = heap_caps_malloc(VOCAB_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s_indices = heap_caps_malloc(VOCAB_SIZE * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_sampling_buffer || !s_softmax_buffer || !s_indices) {
        if (s_sampling_buffer) heap_caps_free(s_sampling_buffer);
        if (s_softmax_buffer) heap_caps_free(s_softmax_buffer);
        if (s_indices) heap_caps_free(s_indices);
        s_sampling_buffer = NULL; s_softmax_buffer = NULL; s_indices = NULL;
        ESP_LOGW(TAG, "Sampling buffers allocation failed; will fall back to slower path");
        return false;
    }
    return true;
}

// Helper: swap indices
static void swap_u16(uint16_t* a, uint16_t* b) { uint16_t t = *a; *a = *b; *b = t; }

uint16_t sample_token(const float* logits, const inference_config_t* config) {
    // Ensure buffers
    bool have_buffers = ensure_sampling_buffers();

    // Temperature scaling
    float temp_scale = 1.0f / config->temperature;

    if (have_buffers) {
        for (int i = 0; i < VOCAB_SIZE; i++) s_sampling_buffer[i] = logits[i] * temp_scale;
    } else {
        // Fallback: operate in-place on a small stack buffer if possible (not recommended)
    }

    // Sampling strategies
    if (config->top_k > 0) {
        // Partial top-k selection without global sort: keep an array of top_k indices
        int k = config->top_k;
        if (k > VOCAB_SIZE) k = VOCAB_SIZE;

        // Initialize top_k with first k indices
        for (int i = 0; i < k; i++) {
            s_indices[i] = (uint16_t)i;
        }
        // Find current minimum in top_k
        int min_idx = 0;
        float min_val = s_sampling_buffer[0];
        for (int i = 1; i < k; i++) {
            float v = s_sampling_buffer[i];
            if (v < min_val) { min_val = v; min_idx = i; }
        }

        // Iterate remaining vocab and maintain top_k
        for (int i = k; i < VOCAB_SIZE; i++) {
            float v = s_sampling_buffer[i];
            if (v > min_val) {
                s_indices[min_idx] = (uint16_t)i;
                // find new min in top_k
                min_idx = 0;
                min_val = s_sampling_buffer[s_indices[0]];
                for (int j = 1; j < k; j++) {
                    float vv = s_sampling_buffer[s_indices[j]];
                    if (vv < min_val) { min_val = vv; min_idx = j; }
                }
            }
        }

        // Randomly choose one index from top_k uniformly (or you could weight by logits)
        uint16_t chosen = s_indices[rand() % k];
        return chosen;
    } else if (config->top_p > 0.0f) {
        // Compute softmax into s_softmax_buffer
        // For numerical stability, subtract max
        float maxv = s_sampling_buffer[0];
        for (int i = 1; i < VOCAB_SIZE; i++) if (s_sampling_buffer[i] > maxv) maxv = s_sampling_buffer[i];
        double sum = 0.0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            float e = expf(s_sampling_buffer[i] - maxv);
            s_softmax_buffer[i] = e;
            sum += e;
        }
        if (sum <= 0.0) {
            // fallback to argmax
            int imax = 0; float mv = s_sampling_buffer[0];
            for (int i = 1; i < VOCAB_SIZE; i++) if (s_sampling_buffer[i] > mv) { mv = s_sampling_buffer[i]; imax = i; }
            return imax;
        }
        // Normalize into probabilities
        for (int i = 0; i < VOCAB_SIZE; i++) s_softmax_buffer[i] /= (float)sum;

        // Prepare indices
        for (int i = 0; i < VOCAB_SIZE; i++) s_indices[i] = (uint16_t)i;

        // Sort indices by probability descending using qsort with a simple comparator
        // Use an array of pointers or indices; implement comparator that uses s_softmax_buffer
        int cmpfunc(const void* a, const void* b) {
            uint16_t ia = *(const uint16_t*)a;
            uint16_t ib = *(const uint16_t*)b;
            float va = s_softmax_buffer[ia];
            float vb = s_softmax_buffer[ib];
            if (va < vb) return 1;
            if (va > vb) return -1;
            return 0;
        }
        qsort(s_indices, VOCAB_SIZE, sizeof(uint16_t), cmpfunc);

        // Find cutoff where cumulative prob >= top_p
        float cum = 0.0f;
        int cutoff = VOCAB_SIZE;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            cum += s_softmax_buffer[s_indices[i]];
            if (cum >= config->top_p) { cutoff = i + 1; break; }
        }
        if (cutoff <= 0) cutoff = 1;

        // Sample from the cutoff using probabilities
        float r = ((float)rand() / (float)RAND_MAX) * cum;
        float acc = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            acc += s_softmax_buffer[s_indices[i]];
            if (r <= acc) return s_indices[i];
        }
        return s_indices[cutoff - 1];
    } else {
        // Greedy argmax
        int imax = 0; float mv = s_sampling_buffer[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (s_sampling_buffer[i] > mv) { mv = s_sampling_buffer[i]; imax = i; }
        }
        return (uint16_t)imax;
    }
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