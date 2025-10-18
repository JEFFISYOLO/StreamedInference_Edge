#include "tokenizer.h"
#include "weight_loader.h"
#include "tinyllm_inference.h"
#include "esp_log.h"
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/stat.h>

static const char* TAG = "tokenizer";

// Helper to read entire file into heap buffer
static char* read_file_into_buffer(const char* path, size_t* out_size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    
    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* buf = malloc(sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    
    size_t read = fread(buf, 1, sz, fp);
    fclose(fp);
    
    if (read != sz) { free(buf); return NULL; }
    buf[sz] = '\0';
    if (out_size) *out_size = sz;
    return buf;
}

// Simple JSON parser to extract vocab from large tokenizer.json
static int parse_vocab_size_from_json(const char* buf) {
    // Look for "vocab_size": <number>
    const char* p = strstr(buf, "\"vocab_size\"");
    if (!p) return -1;
    
    p = strchr(p, ':');
    if (!p) return -1;
    p++;
    
    while (*p && (*p == ' ' || *p == '\t')) p++;
    return atoi(p);
}

esp_err_t tokenizer_init(tokenizer_t* tokenizer, const char* vocab_file) {
    ESP_LOGI(TAG, "Initializing tokenizer with vocab file: %s", vocab_file);
    if (!tokenizer || !vocab_file) return ESP_ERR_INVALID_ARG;

    memset(tokenizer, 0, sizeof(tokenizer_t));
    
    // Default special token IDs (common for SentencePiece)
    tokenizer->bos_token_id = 1;
    tokenizer->eos_token_id = 2;
    tokenizer->pad_token_id = 0;
    tokenizer->unk_token_id = 0;

    // Try to read tokenizer_config.json first (it's small)
    const char* sel = weight_loader_get_selected_dir();
    char pathbuf[256];
    
    // First, try tokenizer_config.json for basic info
    snprintf(pathbuf, sizeof(pathbuf), "%s/tokenizer_config.json", sel);
    size_t fsize = 0;
    char* buf = read_file_into_buffer(pathbuf, &fsize);
    
    if (buf) {
        ESP_LOGI(TAG, "Loaded tokenizer config from: %s", pathbuf);
        
        // Parse vocab_size
        int vocab_size = parse_vocab_size_from_json(buf);
        if (vocab_size > 0) {
            tokenizer->vocab_size = vocab_size;
            ESP_LOGI(TAG, "Vocab size from config: %d", vocab_size);
        } else {
            // tokenizer_config.json exists but doesn't contain a numeric vocab_size.
            // Use a safe default to avoid leaving vocab_size == 0 which breaks LM head logic.
            tokenizer->vocab_size = 32000;
            ESP_LOGW(TAG, "tokenizer_config.json missing numeric vocab_size, using default %d", tokenizer->vocab_size);
        }
        
        // Parse special tokens
        const char* p;
        if ((p = strstr(buf, "\"bos_token_id\":"))) {
            tokenizer->bos_token_id = atoi(strchr(p, ':') + 1);
        }
        if ((p = strstr(buf, "\"eos_token_id\":"))) {
            tokenizer->eos_token_id = atoi(strchr(p, ':') + 1);
        }
        if ((p = strstr(buf, "\"pad_token_id\":"))) {
            tokenizer->pad_token_id = atoi(strchr(p, ':') + 1);
        }
        if ((p = strstr(buf, "\"unk_token_id\":"))) {
            tokenizer->unk_token_id = atoi(strchr(p, ':') + 1);
        }
        
        free(buf);
    } else {
        ESP_LOGW(TAG, "Could not load tokenizer_config.json, using defaults");
        tokenizer->vocab_size = 32000; // Default for LLaMA-style models
    }

    // Note: We skip loading the full vocab mapping because:
    // 1. tokenizer.json is 3.6MB (too large for ESP32 RAM)
    // 2. For inference, we only need to decode token IDs back to text
    // 3. We can use a simple byte-level fallback for decoding
    
    ESP_LOGI(TAG, "Tokenizer initialized: vocab_size=%d, bos=%d, eos=%d, pad=%d, unk=%d",
             tokenizer->vocab_size, tokenizer->bos_token_id, 
             tokenizer->eos_token_id, tokenizer->pad_token_id, tokenizer->unk_token_id);

    return ESP_OK;
}

void tokenizer_deinit(tokenizer_t* tokenizer) {
    if (!tokenizer) return;
    if (tokenizer->vocab_tokens) { 
        free(tokenizer->vocab_tokens); 
        tokenizer->vocab_tokens = NULL; 
    }
    if (tokenizer->vocab_strings) {
        for (int i = 0; i < tokenizer->vocab_size; i++) {
            if (tokenizer->vocab_strings[i]) free(tokenizer->vocab_strings[i]);
        }
        free(tokenizer->vocab_strings);
        tokenizer->vocab_strings = NULL;
    }
    ESP_LOGI(TAG, "Tokenizer deinitialized");
}

// Simple word-level tokenizer for testing
// In production, you'd use a proper BPE/SentencePiece tokenizer
esp_err_t tokenizer_encode(tokenizer_t* tokenizer, const char* text, 
                          uint16_t* tokens, int max_tokens, int* actual_tokens) {
    if (!tokenizer || !text || !tokens || !actual_tokens) return ESP_ERR_INVALID_ARG;
    
    ESP_LOGI(TAG, "Encoding text: '%s'", text);
    
    // Simple character-to-token mapping as fallback
    // In a real implementation, you would:
    // 1. Use the tokenizer.model file with SentencePiece library
    // 2. Or pre-tokenize on PC and just pass token IDs to ESP32
    
    int t = 0;
    const unsigned char* ptr = (const unsigned char*)text;
    
    // For now, map each ASCII character to a token ID
    // This is a placeholder - ideally you should tokenize on PC first
    while (*ptr && t < max_tokens - 1) {
        // Map printable ASCII to token range
        // This won't work well but demonstrates the flow
        if (*ptr >= 32 && *ptr < 127) {
            // Map to token IDs in a simple way
            // A real tokenizer would use BPE merges
            tokens[t++] = (*ptr - 32) + 1000; // Arbitrary offset
        } else if (*ptr == ' ') {
            tokens[t++] = 220; // Common space token in many vocabularies
        }
        ptr++;
    }
    
    *actual_tokens = t;
    
    ESP_LOGI(TAG, "Encoded into %d tokens", t);
    for (int i = 0; i < t && i < 10; i++) {
        ESP_LOGI(TAG, "  Token[%d]: %d", i, tokens[i]);
    }
    
    return ESP_OK;
}

// Simple decoder - maps token IDs to characters
esp_err_t tokenizer_decode(tokenizer_t* tokenizer, const uint16_t* tokens, 
                          int num_tokens, char* output, int max_output_len) {
    if (!tokenizer || !tokens || !output) return ESP_ERR_INVALID_ARG;
    
    int out = 0;
    
    for (int i = 0; i < num_tokens && out < max_output_len - 1; i++) {
        uint16_t id = tokens[i];
        
        ESP_LOGD(TAG, "Decoding token[%d]: %d", i, id);
        
        // Skip special tokens
        if (id == tokenizer->bos_token_id || 
            id == tokenizer->eos_token_id || 
            id == tokenizer->pad_token_id) {
            ESP_LOGD(TAG, "  Skipping special token");
            continue;
        }
        
        // Simple reverse mapping for our encoder
        if (id >= 1000 && id < 1095) {
            char c = (id - 1000) + 32;
            output[out++] = c;
        } else if (id == 220) {
            output[out++] = ' ';
        } else if (id < 256) {
            // Byte-level fallback
            output[out++] = (char)id;
        } else {
            // Unknown token - use placeholder
            output[out++] = '?';
        }
    }
    
    output[out] = '\0';
    ESP_LOGI(TAG, "Decoded to: '%s'", output);
    
    return ESP_OK;
}

uint16_t tokenizer_get_bos_token(tokenizer_t* tokenizer) { 
    return tokenizer ? tokenizer->bos_token_id : 1; 
}

uint16_t tokenizer_get_eos_token(tokenizer_t* tokenizer) { 
    return tokenizer ? tokenizer->eos_token_id : 2; 
}

uint16_t tokenizer_get_pad_token(tokenizer_t* tokenizer) { 
    return tokenizer ? tokenizer->pad_token_id : 0; 
}

uint16_t tokenizer_get_unk_token(tokenizer_t* tokenizer) { 
    return tokenizer ? tokenizer->unk_token_id : 0; 
}