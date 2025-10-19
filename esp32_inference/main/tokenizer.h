#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#define MAX_OUTPUT_LEN 2048  // Maximum length for decoded output string

// Simple tokenizer interface
typedef struct {
    uint16_t* vocab_tokens;
    char** vocab_strings;
    int vocab_size;
    uint16_t bos_token_id;
    uint16_t eos_token_id;
    uint16_t pad_token_id;
    uint16_t unk_token_id;
} tokenizer_t;

// Function prototypes
esp_err_t tokenizer_init(tokenizer_t* tokenizer, const char* vocab_file);
void tokenizer_deinit(tokenizer_t* tokenizer);

// Encoding/decoding functions
esp_err_t tokenizer_encode(tokenizer_t* tokenizer, const char* text, 
                          uint16_t* tokens, int max_tokens, int* actual_tokens);
esp_err_t tokenizer_decode(tokenizer_t* tokenizer, const uint16_t* tokens, 
                          int num_tokens, char* output, int max_output_len);

// Utility functions
uint16_t tokenizer_get_bos_token(tokenizer_t* tokenizer);
uint16_t tokenizer_get_eos_token(tokenizer_t* tokenizer);
uint16_t tokenizer_get_pad_token(tokenizer_t* tokenizer);
uint16_t tokenizer_get_unk_token(tokenizer_t* tokenizer);

#endif // TOKENIZER_H

