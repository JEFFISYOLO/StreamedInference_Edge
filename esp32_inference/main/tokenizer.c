#include "tokenizer.h"
#include "weight_loader.h"
#include "tinyllm_inference.h"
#include "esp_log.h"
#include "esp_vfs.h"
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/stat.h>

static const char* TAG = "tokenizer";

// Minimal JSON vocab loader: scans tokenizer.json for the "model"->"vocab" map
// This is a lightweight parser tailored to the tokenizer.json format produced by common tokenizers.

// Helper to read entire file into heap buffer
static char* read_file_into_buffer(const char* path, size_t* out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    size_t sz = st.st_size;
    char* buf = malloc(sz + 1);
    if (!buf) { close(fd); return NULL; }
    ssize_t r = read(fd, buf, sz);
    close(fd);
    if (r != sz) { free(buf); return NULL; }
    buf[sz] = '\0';
    if (out_size) *out_size = sz;
    return buf;
}

esp_err_t tokenizer_init(tokenizer_t* tokenizer, const char* vocab_file) {
    ESP_LOGI(TAG, "Initializing tokenizer with vocab file: %s", vocab_file);
    if (!tokenizer || !vocab_file) return ESP_ERR_INVALID_ARG;

    memset(tokenizer, 0, sizeof(tokenizer_t));
    tokenizer->bos_token_id = 1;
    tokenizer->eos_token_id = 2;
    tokenizer->pad_token_id = 0;
    tokenizer->unk_token_id = 0; // map unknown to <unk> id 0 if not present

    size_t fsize = 0;
    char* buf = read_file_into_buffer(vocab_file, &fsize);
    if (!buf) {
        ESP_LOGW(TAG, "Failed to read tokenizer file %s; trying common candidate paths...", vocab_file);
        // Try candidate paths under the selected weights directory and a few common names
        const char* sel = weight_loader_get_selected_dir();
        const char* candidates[] = {
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.json", // duplicate intentionally for simple fallback
            NULL
        };
        char pathbuf[192];
        for (int i = 0; candidates[i] != NULL; i++) {
            snprintf(pathbuf, sizeof(pathbuf), "%s/%s", sel ? sel : "", candidates[i]);
            buf = read_file_into_buffer(pathbuf, &fsize);
            if (buf) {
                ESP_LOGI(TAG, "Loaded tokenizer file from %s", pathbuf);
                break;
            } else {
                ESP_LOGD(TAG, "Candidate tokenizer path not found: %s", pathbuf);
            }
        }
        if (!buf) {
            ESP_LOGW(TAG, "No tokenizer file found in candidate paths; falling back to simple tokenizer");
            tokenizer->vocab_size = VOCAB_SIZE;
            return ESP_OK;
        }
    }

    // Find "\"vocab\"\s*:\s*{"
    char* p = strstr(buf, "\"vocab\"");
    if (!p) { free(buf); ESP_LOGW(TAG, "No vocab map found in tokenizer.json"); tokenizer->vocab_size = VOCAB_SIZE; return ESP_OK; }
    p = strchr(p, '{');
    if (!p) { free(buf); tokenizer->vocab_size = VOCAB_SIZE; return ESP_OK; }
    p++; // inside object

    // We'll do a first pass to count entries and max id
    int max_id = -1;
    char* q = p;
    while (1) {
        // Find next quoted token string
        char* key_start = strstr(q, "\"");
        if (!key_start) break;
        // ensure it's before a ':'
        char* colon = strchr(key_start + 1, ':');
        if (!colon) break;
        // find closing quote
        char* key_end = strchr(key_start + 1, '"');
        if (!key_end || key_end > colon) { q = colon ? colon + 1 : key_start + 1; continue; }
        // parse id after colon
        char* nump = colon + 1;
        while (*nump && (*nump == ' ' || *nump == '\t')) nump++;
        int id = atoi(nump);
        if (id > max_id) max_id = id;
        q = colon + 1;
    }

    if (max_id < 0) {
        free(buf);
        tokenizer->vocab_size = VOCAB_SIZE;
        ESP_LOGW(TAG, "No vocab entries parsed; using fallback vocab_size=%d", tokenizer->vocab_size);
        return ESP_OK;
    }

    int vocab_count = max_id + 1;
    tokenizer->vocab_size = vocab_count;

    // Allocate vocab_strings array in heap (small)
    tokenizer->vocab_strings = malloc(sizeof(char*) * vocab_count);
    if (!tokenizer->vocab_strings) { free(buf); return ESP_ERR_NO_MEM; }
    for (int i = 0; i < vocab_count; i++) tokenizer->vocab_strings[i] = NULL;

    // Second pass: extract token string keys and assign to id
    q = p;
    while (1) {
        char* key_quote = strchr(q, '"');
        if (!key_quote) break;
        // ensure key_quote precedes a ':'
        char* key_end = strchr(key_quote + 1, '"');
        if (!key_end) break;
        char* colon = strchr(key_end + 1, ':');
        if (!colon) break;
        // Extract key string
        int key_len = key_end - (key_quote + 1);
        if (key_len > 0 && key_len < 256) {
            char keybuf[256];
            memcpy(keybuf, key_quote + 1, key_len);
            keybuf[key_len] = '\0';
            // parse id
            char* nump = colon + 1;
            while (*nump && (*nump == ' ' || *nump == '\t')) nump++;
            int id = atoi(nump);
            if (id >= 0 && id < vocab_count) {
                // store a copy of keybuf into vocab_strings[id]
                tokenizer->vocab_strings[id] = strdup(keybuf);
            }
        }
        q = colon + 1;
    }

    free(buf);
    ESP_LOGI(TAG, "Loaded tokenizer vocab entries: %d (vocab_size=%d)", vocab_count, tokenizer->vocab_size);
    return ESP_OK;
}

void tokenizer_deinit(tokenizer_t* tokenizer) {
    if (!tokenizer) return;
    if (tokenizer->vocab_tokens) { free(tokenizer->vocab_tokens); tokenizer->vocab_tokens = NULL; }
    if (tokenizer->vocab_strings) {
        for (int i = 0; i < tokenizer->vocab_size; i++) {
            if (tokenizer->vocab_strings[i]) free(tokenizer->vocab_strings[i]);
        }
        free(tokenizer->vocab_strings);
        tokenizer->vocab_strings = NULL;
    }
    ESP_LOGI(TAG, "Tokenizer deinitialized");
}

esp_err_t tokenizer_encode(tokenizer_t* tokenizer, const char* text, uint16_t* tokens, int max_tokens, int* actual_tokens) {
    if (!tokenizer || !text || !tokens || !actual_tokens) return ESP_ERR_INVALID_ARG;
    // Simple byte-fallback encoder: split into bytes and map to existing vocab entries if available
    int t = 0;
    const unsigned char* s = (const unsigned char*)text;
    while (*s && t < max_tokens) {
        unsigned char b = *s;
        // Try to find token whose string equals a byte literal like "<0x20>"
        char byte_key[16];
        snprintf(byte_key, sizeof(byte_key), "<0x%02X>", b);
        // Linear search in vocab_strings for that key
        int found = -1;
        if (tokenizer->vocab_strings) {
            for (int i = 0; i < tokenizer->vocab_size; i++) {
                if (tokenizer->vocab_strings[i] && strcmp(tokenizer->vocab_strings[i], byte_key) == 0) {
                    found = i; break;
                }
            }
        }
        if (found >= 0) tokens[t++] = (uint16_t)found;
        else tokens[t++] = tokenizer->unk_token_id;
        s++;
    }
    *actual_tokens = t;
    return ESP_OK;
}

esp_err_t tokenizer_decode(tokenizer_t* tokenizer, const uint16_t* tokens, int num_tokens, char* output, int max_output_len) {
    if (!tokenizer || !tokens || !output) return ESP_ERR_INVALID_ARG;
    int out = 0;
    for (int i = 0; i < num_tokens && out < max_output_len - 1; i++) {
        uint16_t id = tokens[i];
        if (id == tokenizer->bos_token_id || id == tokenizer->eos_token_id || id == tokenizer->pad_token_id) continue;
        if (id < tokenizer->vocab_size && tokenizer->vocab_strings && tokenizer->vocab_strings[id]) {
            const char* tk = tokenizer->vocab_strings[id];
            // Check if token is in the form <0xHH>
            if (tk[0] == '<' && tk[1] == '0' && tk[2] == 'x') {
                unsigned int val = 0;
                if (sscanf(tk + 3, "%2X", &val) == 1) {
                    output[out++] = (char)val;
                    continue;
                }
            }
            // Otherwise copy printable token bytes (best effort)
            int len = strlen(tk);
            for (int j = 0; j < len && out < max_output_len - 1; j++) {
                if (out < max_output_len - 1) output[out++] = tk[j];
            }
        } else {
            // fallback: put a placeholder
            if (out < max_output_len - 4) {
                output[out++] = '?';
            }
        }
    }
    output[out] = '\0';
    return ESP_OK;
}

uint16_t tokenizer_get_bos_token(tokenizer_t* tokenizer) { return tokenizer ? tokenizer->bos_token_id : 1; }
uint16_t tokenizer_get_eos_token(tokenizer_t* tokenizer) { return tokenizer ? tokenizer->eos_token_id : 2; }
uint16_t tokenizer_get_pad_token(tokenizer_t* tokenizer) { return tokenizer ? tokenizer->pad_token_id : 0; }
uint16_t tokenizer_get_unk_token(tokenizer_t* tokenizer) { return tokenizer ? tokenizer->unk_token_id : 0; }

