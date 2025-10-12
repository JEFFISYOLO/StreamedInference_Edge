# Tiny-LLM Float32 ESP32-CAM Analysis

## Overview

The Tiny-LLM model has been prepared for ESP32-CAM in float32 format without quantization. Here's the comprehensive analysis and recommendations.

## Model Size Analysis

### 📊 **Size Breakdown**
- **Total Model Size**: 49.55 MB
- **Largest Tensors**:
  - Embeddings: 23.44 MB (6,144,000 parameters)
  - LM Head: 23.44 MB (6,144,000 parameters)
  - MLP layers: ~768 KB each (196,608 parameters)
  - Attention layers: ~144 KB each (36,864 parameters)

### 🎯 **Memory Constraints**
- **ESP32-CAM PSRAM**: 4 MB
- **ESP32-CAM SRAM**: 520 KB
- **Model vs. Available**: 49.55 MB vs 4 MB = **12.4x larger**

## ⚠️ **Critical Challenge**

The float32 model is **12.4x larger** than the available PSRAM, making it impossible to load entirely into memory. This requires aggressive streaming and chunking strategies.

## 🚀 **Streaming Strategy for Float32**

### **1. Embeddings (23.44 MB)**
- **Strategy**: Load per token (768 bytes per token)
- **RAM Usage**: 768 bytes per embedding lookup
- **SD Card I/O**: 1 seek + read per token

### **2. LM Head (23.44 MB)**
- **Strategy**: Chunked loading in 256-token batches
- **Chunk Size**: 256 × 192 × 4 = 196,608 bytes (~192 KB)
- **Number of Chunks**: 125 chunks (32,000 ÷ 256)
- **RAM Usage**: 192 KB per chunk + 64 KB for logits buffer

### **3. MLP Layers (768 KB each)**
- **Strategy**: Load sequentially, process in smaller chunks
- **Chunk Size**: 128 dimensions × 192 × 4 = 98,304 bytes (~96 KB)
- **Number of Chunks**: 8 chunks per layer
- **RAM Usage**: 96 KB per chunk + working buffers

### **4. Attention Layers (144 KB each)**
- **Strategy**: Load sequentially (fits in RAM)
- **RAM Usage**: 144 KB per layer + KV cache

## 📋 **Memory Usage Estimation**

### **Peak RAM Usage**
```
Embedding lookup:     768 bytes
LM Head chunk:        196,608 bytes (192 KB)
MLP chunk:            98,304 bytes (96 KB)
Attention weights:    147,456 bytes (144 KB)
KV Cache:             ~400,000 bytes (400 KB)
Working buffers:      ~100,000 bytes (100 KB)
-------------------------------------------
Total Peak:           ~940 KB
```

### **PSRAM Requirements**
- **Peak Usage**: ~940 KB (fits in 4 MB PSRAM ✅)
- **KV Cache**: 400 KB (grows with sequence length)
- **Working Buffers**: 100 KB

## 🔧 **Implementation Requirements**

### **ESP32 Configuration**
```c
// sdkconfig.defaults additions for float32
CONFIG_SPIRAM_BOOT_INIT=y
CONFIG_SPIRAM_USE_MALLOC=y
CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=0
CONFIG_SPIRAM_MALLOC_RESERVE_INTERNAL=32768
CONFIG_FREERTOS_HZ=1000
CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192
```

### **SD Card Requirements**
- **Minimum Size**: 64 MB (for 49.55 MB model + OS overhead)
- **Speed**: Class 10+ recommended
- **Format**: FAT32

## ⚡ **Performance Expectations**

### **Inference Speed**
- **Target**: 0.1-0.3 tokens/second (slower than quantized)
- **Bottlenecks**:
  - SD card I/O (125 seeks for LM head per token)
  - Float32 arithmetic on ESP32 CPU
  - Large tensor operations

### **Memory Access Pattern**
```
Per Token Generation:
1. Load embedding (1 SD read)
2. Load attention weights (4 SD reads)
3. Process attention
4. Load MLP weights (3 SD reads × 8 chunks)
5. Process MLP
6. Load LM head chunks (125 SD reads)
7. Sample token
```

## 🎯 **Recommendations**

### **✅ Feasible with Optimizations**
The float32 model can run on ESP32-CAM with the following:

1. **Aggressive Chunking**: Process large tensors in small chunks
2. **Sequential Loading**: Load weights only when needed
3. **PSRAM Utilization**: Store working buffers in PSRAM
4. **Fast SD Card**: Use high-speed SD card for I/O

### **⚡ Performance Optimizations**
1. **Cache Frequently Used Weights**: Cache small weights in flash/PSRAM
2. **Optimize Chunk Sizes**: Balance RAM usage vs. SD seeks
3. **Parallel I/O**: Use multiple SD card operations if possible
4. **Float32 Math Optimization**: Use ESP32's FPU if available

### **🔧 Implementation Strategy**
1. **Start with Quantized Version**: Test with int8 first for baseline
2. **Implement Float32 Streaming**: Add float32 support incrementally
3. **Profile Performance**: Measure actual SD I/O and compute times
4. **Optimize Bottlenecks**: Focus on slowest operations

## 📁 **File Structure**

```
esp32_float32_weights/
├── embed_tokens.bin     (23.44 MB) - Input embeddings
├── q_proj.bin          (144 KB)    - Query projection
├── k_proj.bin          (72 KB)     - Key projection  
├── v_proj.bin          (72 KB)     - Value projection
├── o_proj.bin          (144 KB)    - Output projection
├── gate_proj.bin       (768 KB)    - MLP gate projection
├── up_proj.bin         (768 KB)    - MLP up projection
├── down_proj.bin       (768 KB)    - MLP down projection
├── lm_head.bin         (23.44 MB)  - Language modeling head
├── norm1.bin           (0.8 KB)    - Layer norm 1 parameters
├── norm2.bin           (0.8 KB)    - Layer norm 2 parameters
├── final_norm.bin      (0.8 KB)    - Final layer norm
└── model_config.json   (config)    - Model configuration
```

**Total Size: 49.55 MB**

## 🚀 **Next Steps**

1. **✅ Float32 Weights Prepared** - Ready for ESP32 deployment
2. **🔄 Implement Chunked Loading** - Update ESP32 code for aggressive chunking
3. **🔄 Test Memory Usage** - Verify peak RAM usage fits in PSRAM
4. **🔄 Performance Testing** - Measure actual inference speed
5. **🔄 Optimization** - Fine-tune chunk sizes and I/O patterns

## 💡 **Alternative Approaches**

If performance is insufficient:

1. **Hybrid Quantization**: Use int8 for large tensors, float32 for small ones
2. **Model Compression**: Use knowledge distillation to create smaller model
3. **Hardware Upgrade**: Consider ESP32-S3 with more PSRAM
4. **Cloud Offloading**: Use ESP32 for preprocessing, cloud for inference

The float32 Tiny-LLM model is **technically feasible** on ESP32-CAM with aggressive streaming, but will be significantly slower than the quantized version. The choice depends on your accuracy vs. performance requirements.

