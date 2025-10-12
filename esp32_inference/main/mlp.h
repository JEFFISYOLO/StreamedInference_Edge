#ifndef MLP_H
#define MLP_H

#include <stdint.h>
#include "tinyllm_inference.h"

// MLP computation structure
typedef struct {
    float gate_output[INTERMEDIATE_SIZE];
    float up_output[INTERMEDIATE_SIZE];
    float swish_output[INTERMEDIATE_SIZE];
    float mlp_output[HIDDEN_SIZE];
} mlp_state_t;

// Function prototypes
esp_err_t compute_mlp_layer(float* hidden_state, mlp_state_t* mlp_state);

// Individual MLP components
esp_err_t compute_gate_projection(float* gate_output, const float* hidden_state);
esp_err_t compute_up_projection(float* up_output, const float* hidden_state);
esp_err_t compute_swish_activation(float* swish_output, const float* gate_output, 
                                  const float* up_output);
esp_err_t compute_down_projection(float* mlp_output, const float* swish_output);

// Chunked computation for large projections
esp_err_t compute_projection_chunked(float* output, const float* input, 
                                   const char* weight_file, int output_size, int input_size);

#endif // MLP_H

