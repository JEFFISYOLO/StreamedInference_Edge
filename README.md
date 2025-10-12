# Tiny-LLM ESP32 Streaming Inference

A lightweight runtime environment for executing the Tiny-LLM transformer model on ESP32-CAM microcontrollers using streaming inference from SD card storage.

## Overview

This project implements a streaming inference system that allows running a 12.8M parameter Tiny-LLM model on ESP32-CAM hardware with limited RAM (~520KB SRAM + 4MB PSRAM). The model weights are stored on an SD card and streamed layer-by-layer during inference to stay within memory constraints.

### Key Features

- **Streaming Inference**: Model weights loaded from SD card on-demand
- **Int8 Quantization**: Reduces model size from ~51MB to ~6.4MB
- **Fixed-Point Math**: Optimized for ESP32 CPU without floating-point overhead
- **KV Caching**: Efficient autoregressive generation with grouped query attention
- **Chunked Processing**: Large tensors processed in chunks to fit RAM
- **Target Performance**: 0.5-1 tokens/second on ESP32-CAM

## Model Architecture

- **Vocabulary**: 32,000 tokens
- **Hidden Size**: 192 dimensions
- **Layers**: 1 transformer layer
- **Attention**: 2 query heads, 1 KV head (Grouped Query Attention)
- **FFN**: 1,024 intermediate size with SwiGLU activation
- **Sequence Length**: Up to 1,024 tokens
- **Parameters**: ~12.8M total

## Hardware Requirements

- ESP32-CAM or ESP32-S module
- SD card (4GB+ recommended)
- External PSRAM (4MB recommended)
- SD card slot with proper wiring

## Software Requirements

- ESP-IDF v4.4 or later
- Python 3.8+ with PyTorch and Transformers
- SD card formatted with FAT32

## Quick Start

### 1. Prepare Model Weights

```bash
# Install Python dependencies
pip install torch transformers safetensors numpy

# Run quantization script
python quantize_model.py
```

This will create an `esp32_weights/` directory with quantized model files:
- `embed_tokens.bin` - Input embeddings (6MB)
- `q_proj.bin`, `k_proj.bin`, `v_proj.bin`, `o_proj.bin` - Attention projections
- `gate_proj.bin`, `up_proj.bin`, `down_proj.bin` - MLP projections  
- `lm_head.bin` - Output layer (6MB)
- `norm1.bin`, `norm2.bin`, `final_norm.bin` - Layer normalization
- `quantization_scales.json` - Quantization parameters

### 2. Setup ESP-IDF Project

```bash
# Clone or copy the esp32_inference directory
cd esp32_inference

# Configure for ESP32-CAM
idf.py set-target esp32s3
idf.py menuconfig

# Set configuration:
# - Enable PSRAM
# - Enable SD card support
# - Set CPU frequency to 240MHz
# - Enable FATFS with long filenames
```

### 3. Transfer Model to SD Card

```bash
# Copy quantized weights to SD card
# Create directory: /sdcard/tinyllm_weights/
# Copy all .bin files and .json files from esp32_weights/
```

### 4. Build and Flash

```bash
# Build the project
idf.py build

# Flash to ESP32-CAM
idf.py flash monitor
```

## Project Structure

```
esp32_inference/
├── main/
│   ├── tinyllm_inference.h/c    # Main inference API
│   ├── weight_loader.h/c        # SD card weight loading
│   ├── fixed_point_math.h/c     # Fixed-point arithmetic kernels
│   ├── attention.h/c            # Self-attention implementation
│   ├── mlp.h/c                  # MLP layer with SwiGLU
│   ├── lm_head.h/c              # LM head and sampling
│   ├── tokenizer.h/c            # Simple tokenizer
│   └── app_main.c               # Main application
├── CMakeLists.txt               # Project configuration
└── sdkconfig.defaults           # ESP-IDF defaults
```

## Usage Example

```c
#include "tinyllm_inference.h"

// Initialize inference engine
tinyllm_init();

// Configure inference parameters
inference_config_t config = {
    .temperature = 0.7f,
    .top_k = 50,
    .top_p = 0.9f,
    .max_new_tokens = 50,
    .use_kv_cache = true
};

// Prepare input tokens
uint16_t prompt_tokens[] = {1, 1234, 5678, 9012}; // BOS + your tokens
uint16_t output_tokens[100];

// Run inference
tinyllm_inference(prompt_tokens, 4, output_tokens, 100, &config);

// Cleanup
tinyllm_deinit();
```

## Memory Management

### RAM Usage Breakdown
- **Hidden States**: 192 × 4 bytes = 768 bytes
- **KV Cache**: 1024 × 96 × 2 × 2 bytes = ~400KB (max sequence)
- **Weight Buffer**: 256KB (largest chunk)
- **Temporary Buffers**: ~50KB
- **Total Peak**: ~700KB (fits in PSRAM)

### Streaming Strategy
1. **Embeddings**: Load per-token from SD card (~192 bytes)
2. **Attention**: Load projections sequentially (~36KB each)
3. **MLP**: Process in 128-dimension chunks (~24KB each)
4. **LM Head**: Process in 256-token chunks (~48KB each)

## Performance Optimization

### Expected Performance
- **Inference Speed**: 0.5-1 tokens/second
- **Memory Usage**: ~700KB peak RAM
- **SD Card I/O**: ~100 seeks per token generation

### Optimization Tips
1. **Enable PSRAM**: Essential for KV cache and large buffers
2. **Use Fast SD Card**: Class 10+ recommended for faster I/O
3. **Optimize Chunk Sizes**: Balance RAM usage vs. SD seeks
4. **Fixed-Point Math**: Avoid floating-point operations
5. **Cache Frequently Used Weights**: Consider caching small weights in flash

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Enable PSRAM in menuconfig
   - Reduce chunk sizes in source code
   - Check available heap with `esp_get_free_heap_size()`

2. **SD Card Not Detected**
   - Verify wiring (CS, MOSI, MISO, CLK)
   - Check SD card format (FAT32)
   - Ensure SD card is properly inserted

3. **Slow Inference**
   - Use faster SD card (Class 10+)
   - Enable CPU frequency scaling to 240MHz
   - Optimize chunk sizes for your hardware

4. **Model Loading Errors**
   - Verify all weight files are present on SD card
   - Check file paths in `weight_loader.h`
   - Ensure proper file permissions

### Debug Information

Enable debug logging to monitor:
- Memory usage during inference
- SD card I/O operations
- Token generation progress
- Performance metrics

```c
// Enable debug logging
esp_log_level_set("*", ESP_LOG_DEBUG);
```

## Limitations

- **Single Layer**: Only supports single-layer transformers
- **Fixed Architecture**: Hardcoded for Tiny-LLM dimensions
- **Simple Tokenizer**: Basic character-based tokenization
- **Limited Sampling**: Simplified top-k/top-p implementation
- **No Batching**: Single-token generation only

## Future Improvements

- [ ] Multi-layer transformer support
- [ ] Configurable model architectures
- [ ] Advanced tokenizer implementation (BPE/WordPiece)
- [ ] Optimized matrix multiplication kernels
- [ ] Dynamic chunk size optimization
- [ ] Model compression techniques (int4 quantization)
- [ ] Flash-based weight caching
- [ ] Multi-threaded inference

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with the original Tiny-LLM model license when using this code.

## Acknowledgments

- Original Tiny-LLM model: [arnir0/Tiny-LLM](https://huggingface.co/arnir0/Tiny-LLM)
- ESP-IDF framework and documentation
- Hugging Face Transformers library

