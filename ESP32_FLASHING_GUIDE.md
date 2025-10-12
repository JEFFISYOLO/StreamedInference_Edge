# ESP32-CAM Tiny-LLM Flashing Guide

## Prerequisites

### 1. Install ESP-IDF
```bash
# Download ESP-IDF v4.4 or later
# Follow official installation guide: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

# On Windows:
# 1. Download ESP-IDF installer from: https://dl.espressif.com/dl/esp-idf/
# 2. Run installer and select ESP-IDF v4.4 or later
# 3. Select ESP32 target

# On Linux/Mac:
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32
. ./export.sh
```

### 2. Hardware Requirements
- ESP32-CAM or ESP32-S module
- SD card (64MB+ recommended for float32, 16MB+ for quantized)
- USB cable for programming
- SD card slot properly wired to ESP32

### 3. SD Card Wiring (ESP32-CAM)
```
ESP32-CAM    SD Card
--------     -------
GPIO14  -->  CLK
GPIO15  -->  CMD  
GPIO2   -->  D0
GPIO4   -->  D1
GPIO12  -->  D2
GPIO13  -->  D3
3.3V    -->  VCC
GND     -->  GND
```

## Project Setup

### 1. Navigate to ESP32 Project
```bash
cd C:\Users\HP\Desktop\Python\ESPLLM\TinyLLM\esp32_inference
```

### 2. Set ESP-IDF Target
```bash
idf.py set-target esp32s3
# For ESP32-CAM, use: idf.py set-target esp32
```

### 3. Configure Project
```bash
idf.py menuconfig
```

**Key Configuration Settings:**
```
# Enable PSRAM
Component config → ESP32S3-Specific → Support for external, SPI-connected PSRAM → Enable
Component config → ESP32S3-Specific → Support for external, SPI-connected PSRAM → SPI PSRAM as main memory

# Enable FATFS
Component config → FAT Filesystem support → Enable

# SD/MMC Configuration
Component config → SD/MMC → Enable SD/MMC host driver
Component config → SD/MMC → Host SPI settings → Default input delay (ps) → 0

# Memory Configuration
Component config → ESP32S3-Specific → CPU frequency → 240 MHz
Component config → ESP32S3-Specific → Support for external, SPI-connected PSRAM → Enable

# Logging
Component config → Log output → Default log verbosity → Info
Component config → Log output → Maximum log verbosity level → Debug

# FreeRTOS
Component config → FreeRTOS → Tick rate (Hz) → 1000
Component config → FreeRTOS → Run FreeRTOS only on first core → Disable
```

## Prepare Model Weights

### For Quantized Model (Recommended)
```bash
# Copy quantized weights to SD card
# Create directory: /sdcard/tinyllm_weights/
# Copy all files from: esp32_weights/
```

### For Float32 Model
```bash
# Copy float32 weights to SD card  
# Create directory: /sdcard/tinyllm_float32_weights/
# Copy all files from: esp32_float32_weights/
```

**SD Card Structure:**
```
/sdcard/
├── tinyllm_weights/           (for quantized model)
│   ├── embed_tokens.bin
│   ├── q_proj.bin
│   ├── k_proj.bin
│   ├── v_proj.bin
│   ├── o_proj.bin
│   ├── gate_proj.bin
│   ├── up_proj.bin
│   ├── down_proj.bin
│   ├── lm_head.bin
│   ├── norm1.bin
│   ├── norm2.bin
│   ├── final_norm.bin
│   ├── model_config.json
│   └── quantization_scales.json
└── tinyllm_float32_weights/   (for float32 model)
    ├── embed_tokens.bin
    ├── q_proj.bin
    ├── k_proj.bin
    ├── v_proj.bin
    ├── o_proj.bin
    ├── gate_proj.bin
    ├── up_proj.bin
    ├── down_proj.bin
    ├── lm_head.bin
    ├── norm1.bin
    ├── norm2.bin
    ├── final_norm.bin
    └── model_config.json
```

## Build and Flash

### 1. Build Project
```bash
idf.py build
```

### 2. Connect ESP32-CAM
- Connect USB cable to ESP32-CAM
- Put ESP32-CAM into programming mode (if required)
- Insert SD card with model weights

### 3. Flash Firmware
```bash
# Flash to ESP32-CAM
idf.py flash

# Flash and monitor serial output
idf.py flash monitor

# Alternative: Flash then monitor separately
idf.py flash
idf.py monitor
```

### 4. Monitor Output
```bash
idf.py monitor
# Press Ctrl+] to exit monitor
```

## Expected Output

### Successful Boot:
```
I (1234) app_main: Tiny-LLM ESP32 Inference Engine Starting
I (1235) app_main: Free heap: 123456 bytes
I (1236) app_main: Initializing SD card
I (1237) app_main: SD card mounted successfully
I (1238) tinyllm_inference: Initializing Tiny-LLM inference engine
I (1239) tinyllm_inference: Tiny-LLM inference engine initialized successfully
I (1240) app_main: Starting inference task
I (1241) tinyllm_inference: Starting inference with prompt length: 4
```

### Inference Output:
```
I (2345) tinyllm_inference: Running inference...
I (2346) tinyllm_inference: Generated 1 tokens
Generated tokens: 1234 5678 9012 
I (2347) tinyllm_inference: Inference completed: 3 tokens in 1234 ms (2.43 tokens/sec)
```

## Troubleshooting

### Common Issues

#### 1. SD Card Not Detected
```
E (1234) weight_loader: Failed to mount filesystem
```
**Solutions:**
- Check SD card wiring
- Verify SD card format (FAT32)
- Ensure SD card is properly inserted
- Check power supply (3.3V stable)

#### 2. Out of Memory
```
E (1234) tinyllm_inference: Failed to allocate memory
```
**Solutions:**
- Enable PSRAM in menuconfig
- Reduce chunk sizes in source code
- Use quantized model instead of float32
- Check available heap: `esp_get_free_heap_size()`

#### 3. Model Files Not Found
```
E (1234) weight_loader: Failed to open embedding file
```
**Solutions:**
- Verify SD card contains model weights
- Check file paths in `weight_loader.h`
- Ensure proper directory structure
- Check file permissions

#### 4. Build Errors
```
error: 'fixed_point_math.h' file not found
```
**Solutions:**
- Use float32 version: change includes to `float32_math.h`
- Update CMakeLists.txt to include correct files
- Clean build: `idf.py fullclean && idf.py build`

### Performance Optimization

#### 1. Increase Inference Speed
- Use quantized model (int8)
- Optimize chunk sizes
- Use faster SD card (Class 10+)
- Enable CPU frequency scaling

#### 2. Reduce Memory Usage
- Enable PSRAM
- Use smaller chunk sizes
- Optimize buffer allocation
- Monitor heap usage

#### 3. Improve Accuracy
- Use float32 model
- Fine-tune quantization parameters
- Increase model precision
- Use better calibration data

## Testing

### 1. Basic Functionality Test
```c
// Test with simple prompt
uint16_t test_prompt[] = {1, 1234, 5678};  // BOS + test tokens
uint16_t output_tokens[10];
tinyllm_inference(test_prompt, 3, output_tokens, 10, &config);
```

### 2. Performance Test
```c
// Measure inference time
int64_t start_time = esp_timer_get_time();
tinyllm_inference(prompt, prompt_len, output, max_output, &config);
int64_t end_time = esp_timer_get_time();
float tokens_per_second = tokens_generated / ((end_time - start_time) / 1000000.0f);
```

### 3. Memory Test
```c
// Check memory usage
ESP_LOGI(TAG, "Free heap: %d bytes", esp_get_free_heap_size());
ESP_LOGI(TAG, "Minimum free heap: %d bytes", esp_get_minimum_free_heap_size());
```

## Next Steps

1. **Test Basic Inference**: Verify model loads and runs
2. **Measure Performance**: Check tokens/second and memory usage
3. **Optimize Parameters**: Tune chunk sizes and buffer allocations
4. **Add Features**: Implement proper tokenization, generation loops
5. **Production Ready**: Add error handling, configuration management

The ESP32-CAM is now ready to run Tiny-LLM inference! 🚀
