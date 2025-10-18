#include "weight_loader.h"
#include "tinyllm_inference.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/spi_common.h"
#include "driver/sdmmc_host.h"
#include "driver/sdspi_host.h"
#include "sdmmc_cmd.h"
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <stdio.h>

static const char* TAG = "weight_loader";
static bool sd_card_mounted = false;
static sdmmc_card_t* s_card = NULL;
// Runtime-selected weights directory on the SD card
static char selected_weights_dir[64] = "";

// --- LM head chunk cache implementation ---
typedef struct {
    int chunk_index;
    bool valid;
    uint32_t last_used; // simple timestamp for LRU
    float* buffer; // pointer to chunk data in PSRAM
} lm_cache_entry_t;

static lm_cache_entry_t* lm_cache = NULL;
static int lm_cache_capacity = 0;
static int lm_cache_hits = 0;
static int lm_cache_misses = 0;
static uint32_t lm_cache_time = 1; // simple incrementing counter

static void lm_cache_free_all(void) {
    if (!lm_cache) return;
    for (int i = 0; i < lm_cache_capacity; i++) {
        if (lm_cache[i].buffer) {
            heap_caps_free(lm_cache[i].buffer);
            lm_cache[i].buffer = NULL;
        }
    }
    heap_caps_free(lm_cache);
    lm_cache = NULL;
    lm_cache_capacity = 0;
}

esp_err_t weight_loader_set_lm_cache_capacity(int capacity) {
    if (capacity < 0) return ESP_ERR_INVALID_ARG;
    // If capacity unchanged, nothing to do
    if (capacity == lm_cache_capacity) return ESP_OK;

    // Free existing cache
    lm_cache_free_all();

    if (capacity == 0) return ESP_OK; // disabled

    // Allocate lm_cache array in internal RAM (small) and each buffer in PSRAM
    lm_cache = heap_caps_malloc(sizeof(lm_cache_entry_t) * capacity, MALLOC_CAP_DEFAULT);
    if (!lm_cache) {
        ESP_LOGE(TAG, "Failed to allocate lm_cache entries");
        return ESP_ERR_NO_MEM;
    }
    memset(lm_cache, 0, sizeof(lm_cache_entry_t) * capacity);

    // Buffers will be allocated lazily on demand when a chunk is loaded
    lm_cache_capacity = capacity;
    lm_cache_hits = 0;
    lm_cache_misses = 0;
    lm_cache_time = 1;
    ESP_LOGI(TAG, "LM head cache initialized with capacity=%d", capacity);
    return ESP_OK;
}

void weight_loader_get_lm_cache_stats(int* hits, int* misses, int* capacity) {
    if (hits) *hits = lm_cache_hits;
    if (misses) *misses = lm_cache_misses;
    if (capacity) *capacity = lm_cache_capacity;
}

// Helper: attempt to find chunk in cache; if found, copy pointer into out_buffer and return true
static bool lm_cache_get_chunk(int chunk_index, float** out_buffer) {
    if (!lm_cache || lm_cache_capacity == 0) return false;
    for (int i = 0; i < lm_cache_capacity; i++) {
        if (lm_cache[i].valid && lm_cache[i].chunk_index == chunk_index) {
            lm_cache[i].last_used = ++lm_cache_time;
            *out_buffer = lm_cache[i].buffer;
            lm_cache_hits++;
            return true;
        }
    }
    lm_cache_misses++;
    return false;
}

// Helper: insert or replace an entry with given chunk_index and data (data buffer must be allocated)
static esp_err_t lm_cache_insert_chunk(int chunk_index, float* data_buffer) {
    if (!lm_cache || lm_cache_capacity == 0) return ESP_OK; // cache disabled; caller retains buffer

    // Find an invalid slot first
    int evict_idx = -1;
    for (int i = 0; i < lm_cache_capacity; i++) {
        if (!lm_cache[i].valid) {
            evict_idx = i;
            break;
        }
    }

    // If no invalid slot, pick least-recently-used
    if (evict_idx == -1) {
        uint32_t min_time = UINT32_MAX;
        for (int i = 0; i < lm_cache_capacity; i++) {
            if (lm_cache[i].last_used < min_time) {
                min_time = lm_cache[i].last_used;
                evict_idx = i;
            }
        }
    }

    // Evict if needed
    if (lm_cache[evict_idx].valid && lm_cache[evict_idx].buffer) {
        heap_caps_free(lm_cache[evict_idx].buffer);
        lm_cache[evict_idx].buffer = NULL;
    }

    lm_cache[evict_idx].chunk_index = chunk_index;
    lm_cache[evict_idx].valid = true;
    lm_cache[evict_idx].last_used = ++lm_cache_time;
    lm_cache[evict_idx].buffer = data_buffer;

    return ESP_OK;
}


const char* weight_loader_get_selected_dir(void) {
    return selected_weights_dir[0] ? selected_weights_dir : WEIGHTS_DIR;
}

// ESP32-CAM SD card pin configuration (same as person_detection project)
#define SDCARD_MOSI_PIN 15
#define SDCARD_MISO_PIN 2
#define SDCARD_SCLK_PIN 14
#define SDCARD_CS_PIN   13

esp_err_t weight_loader_init(void) {
    esp_err_t ret = ESP_OK;
    ESP_LOGI(TAG, "Initializing SD card (SDSPI mode for ESP32-CAM)");

    // Use SDSPI host default for ESP32-CAM (this matches the working person_detection setup)
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();

    // Configure SPI bus (HSPI / SPI2 is commonly used for ESP32-CAM)
    spi_bus_config_t bus_cfg = {
        .mosi_io_num = SDCARD_MOSI_PIN,
        .miso_io_num = SDCARD_MISO_PIN,
        .sclk_io_num = SDCARD_SCLK_PIN,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };

    // Initialize SPI bus on HSPI (SPI2_HOST)
    ret = spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize SPI bus: %s", esp_err_to_name(ret));
        return ret;
    }

    // Configure SDSPI device config
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = SDCARD_CS_PIN;
    // SPI MISO/MOSI/SCK are configured via spi_bus_initialize; SDSPI device config only needs CS

    // Mount filesystem (SDSPI)
    esp_vfs_fat_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };

    const char mount_point[] = SD_MOUNT_POINT;
    ret = esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_config, &mount_config, &s_card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount filesystem. "
                     "If you want the card to be formatted, set format_if_mount_failed = true.");
        } else {
            ESP_LOGE(TAG, "Failed to initialize the card (%s). "
                     "Make sure SD card lines have pull-up resistors in place.", esp_err_to_name(ret));
        }
        // Free the SPI bus we initialized
        spi_bus_free(SPI2_HOST);
        return ret;
    }

    sd_card_mounted = true;
    ESP_LOGI(TAG, "SD card mounted successfully at %s", mount_point);
    ESP_LOGI(TAG, "SD card info: %lluMB", ((uint64_t) s_card->ocr >> 20));
    // Diagnostic: check if expected weight files are present on the card and list the directory
    struct stat st;
    if (stat(WEIGHTS_DIR, &st) == 0 && S_ISDIR(st.st_mode)) {
        ESP_LOGI(TAG, "Weights directory exists: %s", WEIGHTS_DIR);
        DIR* d = opendir(WEIGHTS_DIR);
        if (d) {
            struct dirent* entry;
            ESP_LOGI(TAG, "Listing files in %s:", WEIGHTS_DIR);
            int count = 0;
            while ((entry = readdir(d)) != NULL && count < 200) {
                ESP_LOGI(TAG, "  %s", entry->d_name);
                count++;
            }
            closedir(d);
        } else {
            ESP_LOGW(TAG, "Failed to open weights directory for listing: %s", WEIGHTS_DIR);
        }
    } else {
        ESP_LOGW(TAG, "Weights directory not found on card: %s", WEIGHTS_DIR);
    }

    // Try a few common candidate directories and pick the first that exists
    const char* candidates[] = {
        WEIGHTS_DIR,
        "/sdcard/esp32_float32_weights",
        "/sdcard/esp32_weights",
        NULL
    };

    for (int i = 0; candidates[i] != NULL; i++) {
        struct stat st2;
        if (stat(candidates[i], &st2) == 0 && S_ISDIR(st2.st_mode)) {
            strncpy(selected_weights_dir, candidates[i], sizeof(selected_weights_dir) - 1);
            selected_weights_dir[sizeof(selected_weights_dir) - 1] = '\0';
            ESP_LOGI(TAG, "Selected weights directory: %s", selected_weights_dir);
            break;
        }
    }
    if (selected_weights_dir[0] == '\0') {
        ESP_LOGW(TAG, "No candidate weights directory found; will use default WEIGHTS_DIR macro: %s", WEIGHTS_DIR);
        strncpy(selected_weights_dir, WEIGHTS_DIR, sizeof(selected_weights_dir) - 1);
        selected_weights_dir[sizeof(selected_weights_dir) - 1] = '\0';
    }
    
    return ESP_OK;
}

esp_err_t weight_loader_deinit(void) {
    if (sd_card_mounted) {
        const char mount_point[] = SD_MOUNT_POINT;
        esp_vfs_fat_sdcard_unmount(mount_point, s_card);
        spi_bus_free(SPI2_HOST);
        s_card = NULL;
        sd_card_mounted = false;
        ESP_LOGI(TAG, "SD card unmounted");
    }
    return ESP_OK;
}

esp_err_t load_embedding_row(uint16_t token_id, float* buffer) {
    if (!sd_card_mounted) {
        ESP_LOGE(TAG, "SD card not mounted");
        return ESP_ERR_INVALID_STATE;
    }

    char path[128];
    snprintf(path, sizeof(path), "%s/embed_tokens.bin", selected_weights_dir);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ESP_LOGE(TAG, "Failed to open embedding file: %s (errno=%d: %s)", path, errno, strerror(errno));
        return ESP_FAIL;
    }

    // Calculate offset for token_id row
    off_t offset = token_id * HIDDEN_SIZE * sizeof(float);
    if (lseek(fd, offset, SEEK_SET) < 0) {
        close(fd);
        ESP_LOGE(TAG, "Failed to seek to embedding row");
        return ESP_FAIL;
    }

    // Read embedding row
    ssize_t bytes_read = read(fd, buffer, HIDDEN_SIZE * sizeof(float));
    close(fd);

    if (bytes_read != HIDDEN_SIZE * sizeof(float)) {
        ESP_LOGE(TAG, "Failed to read embedding row");
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t load_layer_norm(const char* filename, float* scale, float* bias) {
    if (!sd_card_mounted) {
        ESP_LOGE(TAG, "SD card not mounted");
        return ESP_ERR_INVALID_STATE;
    }

    char path[128];
    const char* base = strrchr(filename, '/');
    if (base) base++; else base = filename;
    snprintf(path, sizeof(path), "%s/%s", selected_weights_dir, base);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ESP_LOGE(TAG, "Failed to open layer norm file: %s (errno=%d: %s)", path, errno, strerror(errno));
        return ESP_FAIL;
    }

    // Read scale and bias
    float norm_data[2];
    ssize_t bytes_read = read(fd, norm_data, sizeof(norm_data));
    close(fd);

    if (bytes_read != sizeof(norm_data)) {
        ESP_LOGE(TAG, "Failed to read layer norm data");
        return ESP_FAIL;
    }

    *scale = norm_data[0];
    *bias = norm_data[1];
    return ESP_OK;
}

esp_err_t load_projection_matrix(const char* filename, float* buffer) {
    if (!sd_card_mounted) {
        ESP_LOGE(TAG, "SD card not mounted");
        return ESP_ERR_INVALID_STATE;
    }

    char path[128];
    const char* base = strrchr(filename, '/');
    if (base) base++; else base = filename;
    snprintf(path, sizeof(path), "%s/%s", selected_weights_dir, base);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ESP_LOGE(TAG, "Failed to open projection file: %s (errno=%d: %s)", path, errno, strerror(errno));
        return ESP_FAIL;
    }

    // Get file size
    struct stat file_stat;
    if (fstat(fd, &file_stat) < 0) {
        close(fd);
        ESP_LOGE(TAG, "Failed to get file size");
        return ESP_FAIL;
    }

    size_t file_size = file_stat.st_size;
    ssize_t bytes_read = read(fd, buffer, file_size);
    close(fd);

    if (bytes_read != file_size) {
        ESP_LOGE(TAG, "Failed to read projection matrix");
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t load_projection_chunk(const char* filename, float* buffer,
                               int chunk_size, int chunk_index) {
    if (!sd_card_mounted) {
        ESP_LOGE(TAG, "SD card not mounted");
        return ESP_ERR_INVALID_STATE;
    }

    char path[128];
    const char* base = strrchr(filename, '/');
    if (base) base++; else base = filename;
    snprintf(path, sizeof(path), "%s/%s", selected_weights_dir, base);

    int64_t t0 = esp_timer_get_time();
    int fd = open(path, O_RDONLY);
    int64_t t_open = esp_timer_get_time();
    if (fd < 0) {
        ESP_LOGE(TAG, "Failed to open projection file: %s (errno=%d: %s) open_us=%lld", path, errno, strerror(errno), (long long)(t_open - t0));
        return ESP_FAIL;
    }

    // Calculate offset for chunk
    off_t offset = chunk_index * chunk_size * sizeof(float);
    if (lseek(fd, offset, SEEK_SET) < 0) {
        close(fd);
        ESP_LOGE(TAG, "Failed to seek to chunk");
        return ESP_FAIL;
    }

    int64_t t_read_start = esp_timer_get_time();
    ssize_t bytes_read = read(fd, buffer, chunk_size * sizeof(float));
    int64_t t_read_end = esp_timer_get_time();
    close(fd);

    if (bytes_read != chunk_size * sizeof(float)) {
        ESP_LOGE(TAG, "Failed to read projection chunk: %s (bytes_read=%zd expected=%zu)", path, bytes_read, chunk_size * sizeof(float));
        return ESP_FAIL;
    }

    ESP_LOGD(TAG, "Projection chunk %d read: open_us=%lld read_us=%lld bytes=%zd", chunk_index,
             (long long)(t_open - t0), (long long)(t_read_end - t_read_start), bytes_read);

    return ESP_OK;
}

esp_err_t load_lm_head_chunk(float* buffer, int chunk_size, int chunk_index) {
    if (!sd_card_mounted) {
        ESP_LOGE(TAG, "SD card not mounted");
        return ESP_ERR_INVALID_STATE;
    }

    // If cache enabled, check for chunk
    float* cached_ptr = NULL;
    if (lm_cache && lm_cache_capacity > 0) {
        if (lm_cache_get_chunk(chunk_index, &cached_ptr)) {
            // Cached found. Copy cached data into caller's buffer (caller expects contiguous chunk_size floats)
            memcpy(buffer, cached_ptr, chunk_size * sizeof(float));
            ESP_LOGD(TAG, "LM head chunk %d served from cache", chunk_index);
            return ESP_OK;
        }
    }

    // Not in cache: read from SD using existing projection chunk loader
    // For LM head, each token column has HIDDEN_SIZE floats, so request chunk_size * HIDDEN_SIZE floats
    esp_err_t ret = load_projection_chunk(LM_HEAD_FILE, buffer, chunk_size * HIDDEN_SIZE, chunk_index);
    if (ret != ESP_OK) return ret;

    // If cache enabled, allocate PSRAM buffer and copy into cache
    if (lm_cache && lm_cache_capacity > 0) {
        size_t bytes = (size_t)chunk_size * HIDDEN_SIZE * sizeof(float);
        float* cache_buf = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (cache_buf) {
            memcpy(cache_buf, buffer, bytes);
            // Insert into cache (cache will take ownership of cache_buf)
            if (lm_cache_insert_chunk(chunk_index, cache_buf) != ESP_OK) {
                // insertion failed -> free
                heap_caps_free(cache_buf);
            }
        } else {
            ESP_LOGW(TAG, "LM cache: failed to allocate PSRAM buffer for caching chunk %d", chunk_index);
        }
    }

    return ESP_OK;
}
