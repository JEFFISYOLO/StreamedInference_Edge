@echo off
echo ========================================
echo ESP32-CAM Tiny-LLM Flashing Script
echo ========================================

echo.
echo Setting ESP-IDF target to ESP32...
idf.py set-target esp32

echo.
echo Building project...
idf.py build

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo Build successful! Ready to flash.
echo.
echo Make sure:
echo 1. ESP32-CAM is connected via USB
echo 2. SD card with model weights is inserted
echo 3. ESP32-CAM is in programming mode (if required)
echo.
echo Press any key to flash firmware...
pause

echo.
echo Flashing firmware...
idf.py flash

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Flash failed! Check connections.
    pause
    exit /b 1
)

echo.
echo Flashing successful! Starting serial monitor...
echo Press Ctrl+] to exit monitor
echo.
idf.py monitor

echo.
echo Done!
pause
