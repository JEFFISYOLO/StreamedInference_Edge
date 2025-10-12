#include "mlp.h"
#include "weight_loader.h"
#include "float32_math.h"
#include "esp_log.h"
#include <string.h>
#include <stdlib.h>
#include <esp_heap_caps.h>

static const char* TAG = "mlp";

// Temporary buffers for streaming computation
// Use a heap-allocated weight buffer to avoid large .bss
static float* weight_buffer = NULL;

esp_err_t compute_mlp_layer(float* hidden_state, mlp_state_t* mlp_state) {
    ESP_LOGD(TAG, "Computing MLP layer");
    
    // 1. Compute gate projection
    ESP_ERROR_CHECK(compute_gate_projection(mlp_state->gate_output, hidden_state));
    
    // 2. Apply SiLU activation
    silu_float(mlp_state->gate_output, mlp_state->gate_output, INTERMEDIATE_SIZE);
    
    // 3. Compute up projection
    ESP_ERROR_CHECK(compute_up_projection(mlp_state->up_output, hidden_state));
    
    // 4. Elementwise multiplication (SwiGLU)
    ESP_ERROR_CHECK(compute_swish_activation(mlp_state->swish_output, 
                                           mlp_state->gate_output, 
                                           mlp_state->up_output));
    
    // 5. Compute down projection
    ESP_ERROR_CHECK(compute_down_projection(mlp_state->mlp_output, mlp_state->swish_output));
    
    // 6. Residual connection
    vector_add_float(hidden_state, hidden_state, mlp_state->mlp_output, HIDDEN_SIZE);
    
    return ESP_OK;
}

esp_err_t compute_projection_chunked(float* output, const float* input,
                                   const char* weight_file, int output_size, int input_size) {
    int chunk_size = MAX_CHUNK_SIZE / (input_size * sizeof(float));
    if (chunk_size <= 0) chunk_size = 1;
    if (chunk_size > output_size) chunk_size = output_size;
    
    int num_chunks = (output_size + chunk_size - 1) / chunk_size;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_idx = chunk * chunk_size;
        int end_idx = (start_idx + chunk_size < output_size) ? start_idx + chunk_size : output_size;
        int actual_chunk_size = end_idx - start_idx;
        
        // Ensure weight buffer is allocated (PSRAM-capable)
            if (!weight_buffer) {
                weight_buffer = heap_caps_malloc(MAX_CHUNK_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                if (!weight_buffer) {
                    ESP_LOGE(TAG, "Failed to allocate weight_buffer");
                    return ESP_ERR_NO_MEM;
                }
            }

        // Load weight chunk
        esp_err_t ret = load_projection_chunk(weight_file, weight_buffer,
                                            actual_chunk_size, chunk);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to load weight chunk %d", chunk);
            return ret;
        }
        
        // Compute partial output (heap temporary to avoid large stack frames)
        float* chunk_output = heap_caps_malloc(actual_chunk_size * sizeof(float), MALLOC_CAP_DEFAULT);
        if (!chunk_output) {
            ESP_LOGE(TAG, "Failed to allocate chunk_output");
            return ESP_ERR_NO_MEM;
        }
        matvec_float(chunk_output, weight_buffer, input, actual_chunk_size, input_size);
        
        // Copy to final output
        for (int i = 0; i < actual_chunk_size; i++) {
            output[start_idx + i] = chunk_output[i];
        }
    heap_caps_free(chunk_output);
    }
    
    return ESP_OK;
}

esp_err_t compute_gate_projection(float* gate_output, const float* hidden_state) {
    return compute_projection_chunked(gate_output, hidden_state, GATE_PROJ_FILE, 
                                    INTERMEDIATE_SIZE, HIDDEN_SIZE);
}

esp_err_t compute_up_projection(float* up_output, const float* hidden_state) {
    return compute_projection_chunked(up_output, hidden_state, UP_PROJ_FILE, 
                                    INTERMEDIATE_SIZE, HIDDEN_SIZE);
}

esp_err_t compute_swish_activation(float* swish_output, const float* gate_output,
                                 const float* up_output) {
    // Elementwise multiplication: swish = gate * up
    for (int i = 0; i < INTERMEDIATE_SIZE; i++) {
        swish_output[i] = gate_output[i] * up_output[i];
    }
    
    return ESP_OK;
}

esp_err_t compute_down_projection(float* mlp_output, const float* swish_output) {
    return compute_projection_chunked(mlp_output, swish_output, DOWN_PROJ_FILE, 
                                    HIDDEN_SIZE, INTERMEDIATE_SIZE);
}