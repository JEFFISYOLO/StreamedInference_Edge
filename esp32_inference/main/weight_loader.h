#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <stdint.h>
#include "esp_err.h"

// File paths for model weights on SD card
#define SD_MOUNT_POINT "/sdcard"
#define WEIGHTS_DIR "/sdcard/tinyllm_float32_weights"

// Weight file names
#define EMBED_FILE WEIGHTS_DIR "/embed_tokens.bin"
#define NORM1_FILE WEIGHTS_DIR "/norm1.bin"
#define Q_PROJ_FILE WEIGHTS_DIR "/q_proj.bin"
#define K_PROJ_FILE WEIGHTS_DIR "/k_proj.bin"
#define V_PROJ_FILE WEIGHTS_DIR "/v_proj.bin"
#define O_PROJ_FILE WEIGHTS_DIR "/o_proj.bin"
#define NORM2_FILE WEIGHTS_DIR "/norm2.bin"
#define GATE_PROJ_FILE WEIGHTS_DIR "/gate_proj.bin"
#define UP_PROJ_FILE WEIGHTS_DIR "/up_proj.bin"
#define DOWN_PROJ_FILE WEIGHTS_DIR "/down_proj.bin"
#define FINAL_NORM_FILE WEIGHTS_DIR "/final_norm.bin"
#define LM_HEAD_FILE WEIGHTS_DIR "/lm_head.bin"

// Function prototypes
esp_err_t weight_loader_init(void);
esp_err_t weight_loader_deinit(void);

// Return the runtime-selected weights directory (null-terminated C string)
const char* weight_loader_get_selected_dir(void);

// Embedding loading
esp_err_t load_embedding_row(uint16_t token_id, float* buffer);

// Layer norm loading
esp_err_t load_layer_norm(const char* filename, float* scale, float* bias);

// Projection loading (full matrix)
esp_err_t load_projection_matrix(const char* filename, float* buffer);

// Chunked projection loading for large matrices
esp_err_t load_projection_chunk(const char* filename, float* buffer,
                               int chunk_size, int chunk_index);

// LM head chunked loading
esp_err_t load_lm_head_chunk(float* buffer, int chunk_size, int chunk_index);

// LM head chunk cache control (simple LRU cache stored in PSRAM)
// capacity == number of chunks to keep in PSRAM (0 disables cache)
esp_err_t weight_loader_set_lm_cache_capacity(int capacity);
void weight_loader_get_lm_cache_stats(int* hits, int* misses, int* capacity);

#endif // WEIGHT_LOADER_H
