#!/bin/bash

echo "🔧 Whisper Troubleshooting Script"
echo "================================="

# Check system info
echo "📊 System Information:"
echo "OS: $(uname -s) $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Python: $(python3 --version)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}' 2>/dev/null || echo 'N/A')"
echo ""

# Check if we're in the right directory
if [ ! -f "whisper_online.py" ]; then
    echo "❌ whisper_online.py not found in current directory"
    echo "Please run this script from the whisper_streaming directory"
    exit 1
fi

echo "✅ Found whisper_online.py"

# Find and clear model cache
echo "🧹 Clearing Whisper model cache..."

# Common cache locations
CACHE_DIRS=(
    "$HOME/.cache/whisper"
    "$HOME/.cache/huggingface"
    "$HOME/.cache/faster-whisper"
    "$HOME/.cache/whisper-timestamped"
    "/tmp/whisper"
    "$PWD/models"
)

for cache_dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$cache_dir" ]; then
        echo "🗑️  Clearing cache: $cache_dir"
        rm -rf "$cache_dir"
        echo "   ✅ Cleared"
    else
        echo "   ⏭️  Not found: $cache_dir"
    fi
done



# Test different backends and models
MODELS=("tiny.en" "tiny" "base.en")
BACKENDS=("faster-whisper" "whisper_timestamped")

echo ""
echo "🧪 Testing different configurations..."

for backend in "${BACKENDS[@]}"; do
    echo ""
    echo "🔍 Testing backend: $backend"
    
    for model in "${MODELS[@]}"; do
        echo "  🧠 Testing model: $model"
        
        # Set memory limit to prevent crashes
        ulimit -v 4194304  # 4GB virtual memory limit
        
        # Run with timeout to prevent hanging
        timeout 60s python3 whisper_online.py "$TEST_FILE" \
            --model "$model" \
            --backend "$backend" \
            --language en \
            --min-chunk-size 1 \
            --log-level INFO 2>&1 | head -n 20
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "     ✅ SUCCESS: $backend with $model works!"
            WORKING_BACKEND="$backend"
            WORKING_MODEL="$model"
            break 2
        elif [ $exit_code -eq 124 ]; then
            echo "     ⏰ TIMEOUT: $backend with $model (>60s)"
        else
            echo "     ❌ FAILED: $backend with $model (exit code: $exit_code)"
        fi
        
        # Clear any potential memory issues
        python3 -c "import gc; gc.collect()" 2>/dev/null
        sleep 1
    done
done

if [ -n "$WORKING_BACKEND" ]; then
    echo ""
    echo "🎉 Found working configuration!"
    echo "   Backend: $WORKING_BACKEND"
    echo "   Model: $WORKING_MODEL"
    echo ""
    echo "🚀 Recommended command:"
    echo "python3 whisper_online.py your_audio.wav \\"
    echo "    --backend $WORKING_BACKEND \\"
    echo "    --model $WORKING_MODEL \\"
    echo "    --language en \\"
    echo "    --min-chunk-size 1 \\"
    echo "    --vac"
else
    echo ""
    echo "❌ No working configuration found"
    echo ""
    echo "🔧 Additional troubleshooting steps:"
    echo "1. Check system requirements:"
    echo "   - Sufficient RAM (>2GB available)"
    echo "   - Python 3.8+"
    echo "   - All dependencies installed"
    echo ""
    echo "2. Try reinstalling whisper backends:"
    echo "   pip uninstall faster-whisper whisper-timestamped"
    echo "   pip install faster-whisper"
    echo ""
    echo "3. For debugging, run with more verbose output:"
    echo "   python3 whisper_online.py $TEST_FILE --model tiny.en --log-level DEBUG"
    echo ""
    echo "4. Check system logs for more details:"
    echo "   dmesg | tail -20"
fi

# Check for common issues
echo ""
echo "🔍 Checking for common issues..."

# Check available memory
available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}' 2>/dev/null)
if [ -n "$available_mem" ] && [ "$available_mem" -lt 1000 ]; then
    echo "⚠️  Low memory: ${available_mem}MB available (recommend >1GB)"
fi

# Check for conflicting packages
python3 -c "
import sys
try:
    import whisper
    print('⚠️  Found openai-whisper package - may conflict with whisper-online')
    print('   Consider: pip uninstall whisper')
except ImportError:
    print('✅ No conflicting whisper package found')

try:
    import torch
    print('✅ PyTorch available')
except ImportError:
    print('⚠️  PyTorch not found - required for VAC feature')
    print('   Install with: pip install torch torchaudio')
" 2>/dev/null

# Check CUDA availability (if relevant)
python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        print('✅ CUDA available: {} devices'.format(torch.cuda.device_count()))
        print('   GPU Memory: {:.1f}GB'.format(torch.cuda.get_device_properties(0).total_memory / 1e9))
    else:
        print('ℹ️  CUDA not available (using CPU)')
except ImportError:
    pass
" 2>/dev/null

echo ""
echo "🏁 Troubleshooting complete!"

# Clean up test file if we created it
if [ "$TEST_FILE" = "test_audio.wav" ] && [ -f "test_audio.wav" ]; then
    rm test_audio.wav
    echo "🧹 Cleaned up test audio file"
fi