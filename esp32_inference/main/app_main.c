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

// Example inference configuration
static inference_config_t inference_config = {
    .temperature = 0.8f,
    .top_k = 50,
    .top_p = 0.9f,
    .max_new_tokens = 64,
    .use_kv_cache = true
};

void app_main(void) {
    ESP_LOGI(TAG, "Tiny-LLM ESP32-CAM Inference Engine Starting");
    ESP_LOGI(TAG, "Free heap: %" PRIu32 " bytes", esp_get_free_heap_size());
    
    // Initialize Tiny-LLM inference engine
    esp_err_t ret = tinyllm_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize Tiny-LLM: %s", esp_err_to_name(ret));
        return;
    }
    
    ESP_LOGI(TAG, "Tiny-LLM inference engine initialized successfully");
    // Enable LM head cache (tune capacity as needed)
    weight_loader_set_lm_cache_capacity(16);
    
    // Initialize tokenizer and encode a prompt string
    tokenizer_t tokenizer;
    const char* weights_dir = weight_loader_get_selected_dir();
    char vocab_path[128];
    snprintf(vocab_path, sizeof(vocab_path), "%s/tokenizer.json", weights_dir);
    tokenizer_init(&tokenizer, vocab_path);

    const char* prompt_text = "Hello world";
    uint16_t prompt_tokens[256];
    int prompt_len = 0;
    tokenizer_encode(&tokenizer, prompt_text, prompt_tokens, sizeof(prompt_tokens)/sizeof(prompt_tokens[0]), &prompt_len);

    // Output buffer for generated tokens
    const int max_output_len = 128; // request a longer generation
    uint16_t output_tokens[max_output_len];
    memset(output_tokens, 0, sizeof(output_tokens));
    
    ESP_LOGI(TAG, "Starting inference with prompt length: %d", prompt_len);
    
    // Run inference
    ret = tinyllm_inference(prompt_tokens, prompt_len, output_tokens, max_output_len, &inference_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Inference failed: %s", esp_err_to_name(ret));
        return;
    }
    
    ESP_LOGI(TAG, "Inference completed successfully");

    // Print LM cache stats
    int hits = 0, misses = 0, cap = 0;
    weight_loader_get_lm_cache_stats(&hits, &misses, &cap);
    ESP_LOGI(TAG, "LM cache stats: capacity=%d hits=%d misses=%d", cap, hits, misses);
    // Find how many tokens were produced (stop at EOS if present)
    int num_generated = max_output_len;
    uint16_t eos_id = tokenizer_get_eos_token(&tokenizer);
    for (int i = 0; i < max_output_len; i++) {
        if (output_tokens[i] == eos_id) {
            num_generated = i + 1; // include EOS in decoding
            break;
        }
    }

    ESP_LOGI(TAG, "Generated %d tokens (including EOS if present)", num_generated);

    // Decode generated tokens into a readable sentence
    char decoded[1024];
    memset(decoded, 0, sizeof(decoded));
    tokenizer_decode(&tokenizer, output_tokens, num_generated, decoded, sizeof(decoded));
    ESP_LOGI(TAG, "Decoded sentence: %s", decoded);
    
    // Keep the system running
    ESP_LOGI(TAG, "System running. Press reset to run inference again.");
    
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000)); // Sleep for 10 seconds
        ESP_LOGI(TAG, "System alive. Free heap: %" PRIu32 " bytes", esp_get_free_heap_size());
    }
}