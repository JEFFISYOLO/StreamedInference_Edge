#include "tinyllm_inference.h"
#include "weight_loader.h"
#include "attention.h"
#include "mlp.h"
#include "lm_head.h"
#include "float32_math.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <string.h>
#include <stdlib.h>

static const char* TAG = "tinyllm_inference";

// Global model state
static model_state_t g_model_state;
static bool g_model_initialized = false;

// Inference timing
static int64_t g_inference_start_time = 0;
static int g_tokens_generated = 0;

esp_err_t tinyllm_init(void) {
    if (g_model_initialized) {
        ESP_LOGW(TAG, "Model already initialized");
        return ESP_OK;
    }
    
    ESP_LOGI(TAG, "Initializing Tiny-LLM inference engine");
    
    // Initialize weight loader (SD card)
    esp_err_t ret = weight_loader_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize weight loader");
        return ret;
    }
    
    // Initialize model state
    memset(&g_model_state, 0, sizeof(model_state_t));
    // Allocate KV caches in PSRAM to avoid large .bss usage
    g_model_state.kv_cache_k = heap_caps_malloc(KV_CACHE_SIZE * sizeof(hidden_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    g_model_state.kv_cache_v = heap_caps_malloc(KV_CACHE_SIZE * sizeof(hidden_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g_model_state.kv_cache_k || !g_model_state.kv_cache_v) {
        ESP_LOGE(TAG, "Failed to allocate KV caches");
        if (g_model_state.kv_cache_k) heap_caps_free(g_model_state.kv_cache_k);
        if (g_model_state.kv_cache_v) heap_caps_free(g_model_state.kv_cache_v);
        weight_loader_deinit();
        return ESP_ERR_NO_MEM;
    }
    memset(g_model_state.kv_cache_k, 0, KV_CACHE_SIZE * sizeof(hidden_t));
    memset(g_model_state.kv_cache_v, 0, KV_CACHE_SIZE * sizeof(hidden_t));
    g_model_state.cache_valid = false;
    
    g_model_initialized = true;
    ESP_LOGI(TAG, "Tiny-LLM inference engine initialized successfully");
    
    return ESP_OK;
}

esp_err_t tinyllm_inference(const uint16_t* prompt_tokens, int prompt_len,
                           uint16_t* output_tokens, int max_output_len,
                           const inference_config_t* config) {
    if (!g_model_initialized) {
        ESP_LOGE(TAG, "Model not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (prompt_len <= 0 || prompt_len > MAX_SEQ_LEN) {
        ESP_LOGE(TAG, "Invalid prompt length: %d", prompt_len);
        return ESP_ERR_INVALID_ARG;
    }
    
    // Use a simple literal log (avoid complex formatting here to prevent log-time crashes)
    ESP_LOGI(TAG, "Starting inference");
    g_inference_start_time = esp_timer_get_time();
    g_tokens_generated = 0;
    
    // Clear KV cache for new inference
    clear_kv_cache(g_model_state.kv_cache_k, g_model_state.kv_cache_v);
    g_model_state.position = 0;

    // Allocate transformer temporary state on heap to avoid large stack usage
    attention_state_t* attn_state = heap_caps_malloc(sizeof(attention_state_t), MALLOC_CAP_DEFAULT);
    mlp_state_t* mlp_state = heap_caps_malloc(sizeof(mlp_state_t), MALLOC_CAP_DEFAULT);
    if (!attn_state || !mlp_state) {
        ESP_LOGE(TAG, "Failed to allocate transformer temporary state");
        if (attn_state) heap_caps_free(attn_state);
        if (mlp_state) heap_caps_free(mlp_state);
        return ESP_ERR_NO_MEM;
    }
    memset(attn_state, 0, sizeof(attention_state_t));
    memset(mlp_state, 0, sizeof(mlp_state_t));

    // Process prompt tokens
    for (int i = 0; i < prompt_len; i++) {
        ESP_LOGD(TAG, "Processing prompt token %d: %d", i, prompt_tokens[i]);

        // Load embedding for current token
        esp_err_t ret = load_embedding_row(prompt_tokens[i], g_model_state.hidden_state);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to load embedding for token %d", prompt_tokens[i]);
            heap_caps_free(attn_state);
            heap_caps_free(mlp_state);
            return ret;
        }

        // Apply transformer layer
        ret = compute_self_attention(g_model_state.hidden_state, g_model_state.position,
                                   g_model_state.kv_cache_k, g_model_state.kv_cache_v, attn_state);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to compute self-attention for prompt token %d", i);
            heap_caps_free(attn_state);
            heap_caps_free(mlp_state);
            return ret;
        }

        ret = compute_mlp_layer(g_model_state.hidden_state, mlp_state);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to compute MLP layer for prompt token %d", i);
            heap_caps_free(attn_state);
            heap_caps_free(mlp_state);
            return ret;
        }

        g_model_state.position++;
    }
    
    // Generate new tokens
    lm_head_state_t lm_head_state;
    memset(&lm_head_state, 0, sizeof(lm_head_state_t));

    // Allocate transformer temporary states on heap for generation
    attention_state_t* gen_attn_state = heap_caps_malloc(sizeof(attention_state_t), MALLOC_CAP_DEFAULT);
    mlp_state_t* gen_mlp_state = heap_caps_malloc(sizeof(mlp_state_t), MALLOC_CAP_DEFAULT);
    if (!gen_attn_state || !gen_mlp_state) {
        ESP_LOGE(TAG, "Failed to allocate generation temporary state");
        if (gen_attn_state) heap_caps_free(gen_attn_state);
        if (gen_mlp_state) heap_caps_free(gen_mlp_state);
        return ESP_ERR_NO_MEM;
    }
    memset(gen_attn_state, 0, sizeof(attention_state_t));
    memset(gen_mlp_state, 0, sizeof(mlp_state_t));

    for (int new_token_idx = 0; new_token_idx < max_output_len; new_token_idx++) {
        ESP_LOGD(TAG, "Generating token %d", new_token_idx);

        // Compute LM head
        esp_err_t ret = compute_lm_head_chunked(g_model_state.hidden_state, &lm_head_state);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to compute LM head");
            heap_caps_free(gen_attn_state);
            heap_caps_free(gen_mlp_state);
            if (lm_head_state.logits) heap_caps_free(lm_head_state.logits);
            if (lm_head_state.temp_logits) heap_caps_free(lm_head_state.temp_logits);
            return ret;
        }

        // Sample next token
        uint16_t next_token = sample_token(lm_head_state.logits, config);
        output_tokens[new_token_idx] = next_token;
        g_tokens_generated++;

        ESP_LOGD(TAG, "Generated token %d: %d", new_token_idx, next_token);

        // Check for end of sequence
        if (next_token == 2) { // EOS token ID
            ESP_LOGI(TAG, "End of sequence token generated");
            break;
        }

        // Load embedding for next token and continue generation
        ret = load_embedding_row(next_token, g_model_state.hidden_state);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to load embedding for generated token %d", next_token);
            heap_caps_free(gen_attn_state);
            heap_caps_free(gen_mlp_state);
            if (lm_head_state.logits) heap_caps_free(lm_head_state.logits);
            if (lm_head_state.temp_logits) heap_caps_free(lm_head_state.temp_logits);
            return ret;
        }

        // Apply transformer layer using heap-allocated temps
        ret = compute_self_attention(g_model_state.hidden_state, g_model_state.position,
                                   g_model_state.kv_cache_k, g_model_state.kv_cache_v, gen_attn_state);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to compute self-attention for generated token");
            heap_caps_free(gen_attn_state);
            heap_caps_free(gen_mlp_state);
            if (lm_head_state.logits) heap_caps_free(lm_head_state.logits);
            if (lm_head_state.temp_logits) heap_caps_free(lm_head_state.temp_logits);
            return ret;
        }

        ret = compute_mlp_layer(g_model_state.hidden_state, gen_mlp_state);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to compute MLP layer for generated token");
            heap_caps_free(gen_attn_state);
            heap_caps_free(gen_mlp_state);
            if (lm_head_state.logits) heap_caps_free(lm_head_state.logits);
            if (lm_head_state.temp_logits) heap_caps_free(lm_head_state.temp_logits);
            return ret;
        }

        g_model_state.position++;

        // Check sequence length limit
        if (g_model_state.position >= MAX_SEQ_LEN) {
            ESP_LOGW(TAG, "Maximum sequence length reached");
            break;
        }
    }

    // Free generation temporaries
    heap_caps_free(gen_attn_state);
    heap_caps_free(gen_mlp_state);
    if (lm_head_state.logits) heap_caps_free(lm_head_state.logits);
    if (lm_head_state.temp_logits) heap_caps_free(lm_head_state.temp_logits);
    
    // Log inference statistics
    int64_t inference_time = esp_timer_get_time() - g_inference_start_time;
    float tokens_per_second = (float)g_tokens_generated / (inference_time / 1000000.0f);
    
    ESP_LOGI(TAG, "Inference completed: %d tokens in %lld ms (%.2f tokens/sec)",
             g_tokens_generated, inference_time / 1000, tokens_per_second);
    
    return ESP_OK;
}

// Removed process_transformer_layer - now handled inline in inference function

void tinyllm_deinit(void) {
    if (g_model_initialized) {
        weight_loader_deinit();
        // Free KV caches (use heap_caps_free to match heap_caps_malloc)
        if (g_model_state.kv_cache_k) {
            heap_caps_free(g_model_state.kv_cache_k);
            g_model_state.kv_cache_k = NULL;
        }
        if (g_model_state.kv_cache_v) {
            heap_caps_free(g_model_state.kv_cache_v);
            g_model_state.kv_cache_v = NULL;
        }
        g_model_initialized = false;
        ESP_LOGI(TAG, "Tiny-LLM inference engine deinitialized");
    }
}

// Removed fixed-point conversion functions - using float32 directly

