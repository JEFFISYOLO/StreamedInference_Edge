#include "tinyllm_inference.h"
#include "weight_loader.h"
#include "tokenizer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>
#include <inttypes.h>

static const char* TAG = "app_main";

// Pre-tokenized test prompts
// These should be generated using your actual tokenizer on PC
// For now, using minimal prompts to test basic functionality

// Test 1: Just BOS token (minimal test)
const uint16_t test1_tokens[] = {1};  // BOS
const int test1_len = 1;

// Test 2: BOS + a few common token IDs (you need to find actual tokens)
// Common patterns for "Hello" in LLaMA-style tokenizers: ~15043, 22557, etc.
// For now, let's try some low token IDs that are likely to exist
const uint16_t test2_tokens[] = {
    1,      // BOS
    15043,  // Common "Hello" token (example - may need adjustment)
    29871,  // Space character in LLaMA tokenizer
    3186,   // Common "world" token (example - may need adjustment)
};
const int test2_len = 4;

// Test 3: Very simple - just a couple low-numbered tokens
const uint16_t test3_tokens[] = {
    1,      // BOS
    450,    // "the" in many tokenizers
    338,    // "is" in many tokenizers
};
const int test3_len = 3;

// Example inference configuration
static inference_config_t inference_config = {
    .temperature = 1.0f,      // Changed to 1.0 for more deterministic results
    .top_k = 40,              // Reduced for faster sampling
    .top_p = 0.9f,
    .max_new_tokens = 10,     // Very short for testing
    .use_kv_cache = true
};

void app_main(void) {
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "Tiny-LLM ESP32-CAM Inference Engine");
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "Free heap: %" PRIu32 " bytes", esp_get_free_heap_size());
    
    // Initialize Tiny-LLM inference engine
    esp_err_t ret = tinyllm_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize Tiny-LLM: %s", esp_err_to_name(ret));
        return;
    }
    
    ESP_LOGI(TAG, "Tiny-LLM engine initialized");
    
    // Enable larger LM head cache for better performance
    ESP_LOGI(TAG, "Setting LM cache capacity to 64 chunks...");
    weight_loader_set_lm_cache_capacity(64);
    
    // Initialize tokenizer (mainly for decoding)
    tokenizer_t tokenizer;
    const char* weights_dir = weight_loader_get_selected_dir();
    char vocab_path[128];
    snprintf(vocab_path, sizeof(vocab_path), "%s/tokenizer_config.json", weights_dir);
    tokenizer_init(&tokenizer, vocab_path);
    
    ESP_LOGI(TAG, "Tokenizer: vocab_size=%d, bos=%d, eos=%d", 
             tokenizer.vocab_size, tokenizer.bos_token_id, tokenizer.eos_token_id);

    // Select which test to run
    const uint16_t* prompt_tokens = test1_tokens;
    int prompt_len = test1_len;
    const char* test_name = "Test 1: BOS only";
    
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "Running: %s", test_name);
    ESP_LOGI(TAG, "Prompt tokens (%d):", prompt_len);
    for (int i = 0; i < prompt_len; i++) {
        ESP_LOGI(TAG, "  [%d]: %d", i, prompt_tokens[i]);
    }
    ESP_LOGI(TAG, "==============================================");
    
    // Output buffer
    const int max_output_len = 10;
    uint16_t output_tokens[max_output_len];
    memset(output_tokens, 0, sizeof(output_tokens));
    
    // Run inference with timing
    ESP_LOGI(TAG, "Starting inference (max %d new tokens)...", max_output_len);
    int64_t start_time = esp_timer_get_time();
    
    ret = tinyllm_inference(prompt_tokens, prompt_len, output_tokens, max_output_len, &inference_config);
    
    int64_t end_time = esp_timer_get_time();
    float elapsed_sec = (end_time - start_time) / 1000000.0f;
    
    ESP_LOGI(TAG, "==============================================");
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Inference FAILED: %s", esp_err_to_name(ret));
        return;
    }
    
    ESP_LOGI(TAG, "Inference completed in %.1f seconds", elapsed_sec);

    // Cache statistics
    int hits = 0, misses = 0, cap = 0;
    weight_loader_get_lm_cache_stats(&hits, &misses, &cap);
    float hit_rate = (hits + misses > 0) ? (100.0f * hits / (hits + misses)) : 0.0f;
    ESP_LOGI(TAG, "LM Cache: cap=%d, hits=%d, misses=%d (%.1f%% hit rate)", 
             cap, hits, misses, hit_rate);
    
    // Count generated tokens
    int num_generated = 0;
    uint16_t eos_id = tokenizer_get_eos_token(&tokenizer);
    for (int i = 0; i < max_output_len; i++) {
        if (output_tokens[i] == eos_id) {
            num_generated = i + 1;
            ESP_LOGI(TAG, "EOS token found at position %d", i);
            break;
        }
        if (output_tokens[i] == 0) {
            num_generated = i;
            break;
        }
    }
    if (num_generated == 0) num_generated = max_output_len;

    // Show generated tokens
    ESP_LOGI(TAG, "Generated %d tokens:", num_generated);
    for (int i = 0; i < num_generated; i++) {
        ESP_LOGI(TAG, "  Output[%d]: %d", i, output_tokens[i]);
    }
    
    // Performance metrics
    float tokens_per_sec = num_generated / elapsed_sec;
    float sec_per_token = elapsed_sec / num_generated;
    ESP_LOGI(TAG, "Performance: %.3f tokens/sec (%.1f sec/token)", 
             tokens_per_sec, sec_per_token);

    // Try to decode (will likely show garbage without proper tokenizer)
    char decoded[512];
    memset(decoded, 0, sizeof(decoded));
    tokenizer_decode(&tokenizer, output_tokens, num_generated, decoded, sizeof(decoded));
    ESP_LOGI(TAG, "Decoded: '%s'", decoded);
    
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "Free heap after inference: %" PRIu32 " bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "==============================================");
    
    // Keep running
    ESP_LOGI(TAG, "System running. Press reset to restart.");
    
    int cycle = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(30000));  // Every 30 seconds
        cycle++;
        ESP_LOGI(TAG, "[%d] Alive. Heap: %" PRIu32 " bytes", 
                 cycle, esp_get_free_heap_size());
    }
}