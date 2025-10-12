#ifndef LM_HEAD_H
#define LM_HEAD_H

#include <stdint.h>
#include "tinyllm_inference.h"

// LM head computation structure
typedef struct {
    float* logits;           // allocated at runtime to avoid large .bss
    float* temp_logits;      // small temporary buffer, allocated as needed
} lm_head_state_t;

// Function prototypes
esp_err_t compute_lm_head_chunked(float* hidden_state, lm_head_state_t* lm_head_state);

// Individual LM head components
esp_err_t compute_lm_head_chunk(float* chunk_logits, const float* hidden_state,
                               int chunk_size, int chunk_index);
esp_err_t merge_lm_head_chunks(float* final_logits, const float* chunk_logits,
                              int chunk_size, int chunk_index);

// Token sampling
uint16_t sample_token(const float* logits, const inference_config_t* config);
uint16_t sample_top_k_tokens(float* logits, int vocab_size, int top_k);
uint16_t sample_top_p_tokens(float* logits, int vocab_size, float top_p);

#endif // LM_HEAD_H

