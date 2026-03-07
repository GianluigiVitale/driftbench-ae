#!/bin/bash
################################################################################
# Download Required Models for Production Validation (PARALLEL)
# 
# Downloads Llama-3.1-8B-Instruct and LlamaGuard-3-8B in parallel to models/
# Requires: HuggingFace account with Meta model access
################################################################################

set -e

# Check if running from reproduction_scripts directory
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] Please run this script from reproduction_scripts directory"
    exit 1
fi

# Enable fast downloads if hf_transfer available
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create models directory
mkdir -p models

echo "======================================================================"
echo "   DriftBench Production Validation - Parallel Model Download"
echo "======================================================================"
echo ""
echo "This script downloads the required models in PARALLEL:"
echo "  1. Llama-3.1-8B-Instruct (~16GB)"
echo "  2. LlamaGuard-3-8B (~19GB)"
echo ""
echo "Total download size: ~35GB"
echo ""
echo "======================================================================"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "[ERROR] huggingface-cli not found"
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

# Check HuggingFace token (silent check - user should already be logged in)
# if [ -z "$HF_TOKEN" ]; then
#     echo "[WARNING] HF_TOKEN environment variable not set"
#     echo "You may need to login with: huggingface-cli login"
#     echo ""
# fi

# Function to download a model
download_model() {
    local name=$1
    local repo=$2
    local dir=$3
    local extra_args=$4
    
    echo "========================================================================"
    echo "[START] Downloading $name..."
    echo "========================================================================"
    huggingface-cli download $repo \
      --local-dir $dir \
      $extra_args \
      2>/dev/null | grep -v "Traceback" | grep -v "RuntimeError" | grep -v "Fatal Python error" | tee /tmp/download_${name}.log
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "[✓ DONE] $name downloaded successfully"
        echo ""
    else
        echo ""
        echo "[✗ FAILED] $name download failed - check /tmp/download_${name}.log"
        echo ""
    fi
}

# Start parallel downloads (background processes)
echo "Launching 2 parallel downloads..."
echo ""

download_model "llama-3.1-8b" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "models/llama-3.1-8b" \
    --exclude "original/*" --exclude "*.bin" &

download_model "llama-guard-3-8b" \
    "meta-llama/Llama-Guard-3-8B" \
    "models/llama-guard-3-8b" \
    --exclude "original/*" --exclude "*.bin" &

# Wait for all background jobs to complete
echo "Downloads running in parallel..."
echo ""

wait

echo ""
echo "======================================================================"
echo "                Parallel Downloads Completed!"
echo "======================================================================"
echo ""
echo "Model locations:"
echo "  - models/llama-3.1-8b/"
echo "  - models/llama-guard-3-8b/"
echo ""

# Check for successful downloads using folder size (>20GB = success)
echo "Checking download status..."
failed=0
for model in llama-3.1-8b llama-guard-3-8b; do
    size=$(du -s models/${model} 2>/dev/null | awk '{print $1}')
    size_gb=$((size / 1024 / 1024))
    
    if [ $size_gb -gt 20 ]; then
        echo "✓ $model: ${size_gb}GB downloaded successfully"
    else
        echo "✗ $model: Only ${size_gb}GB - download may be incomplete"
        failed=1
    fi
done

if [ $failed -eq 0 ]; then
    echo ""
    echo "✓ All downloads completed successfully!"
else
    echo ""
    echo "⚠ Some downloads incomplete - check sizes above"
fi

echo ""

echo ""
echo "Disk usage:"
du -sh models/* 2>/dev/null || echo "  (run 'du -sh models/*' to check sizes)"
echo ""
echo "Next steps:"
echo "  1. Run H100 experiment: python parallel_controller.py --hardware h100 --workers 1"
echo "  2. Run B200 experiment: python parallel_controller.py --hardware b200 --workers 1"
echo "  3. Compute flip rate: python compute_direct_flip_rate.py"
echo ""
echo "======================================================================"
echo ""
