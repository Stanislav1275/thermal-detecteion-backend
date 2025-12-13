#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v conda &> /dev/null && conda info --envs | grep -q "thermal-detection"; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate thermal-detection
elif [ -d "../backend/venv" ]; then
    source ../backend/venv/bin/activate
fi

MODEL_STD="models/best.pt"
MODEL_FOUND=""

if [ -f "$MODEL_STD" ]; then
    MODEL_FOUND="$MODEL_STD"
else
    RUNS_DIR="training/runs"
    [ ! -d "$RUNS_DIR" ] && RUNS_DIR="runs"
    
    if [ -d "$RUNS_DIR" ]; then
        LATEST_RUN=$(ls -td "$RUNS_DIR"/thermal_detection* 2>/dev/null | head -1)
        if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/weights/best.pt" ]; then
            mkdir -p "$(dirname "$MODEL_STD")"
            cp "$LATEST_RUN/weights/best.pt" "$MODEL_STD"
            MODEL_FOUND="$MODEL_STD"
        fi
    fi
fi

if [ -z "$MODEL_FOUND" ] || [ ! -f "$MODEL_FOUND" ]; then
    echo "❌ Модель не найдена"
    exit 1
fi

echo "✓ Модель: $MODEL_FOUND"
python validate.py --model "$MODEL_FOUND" --data thermal.yaml --output results --conf 0.25 --visualize

if [ -f "results/metrics.json" ]; then
    echo ""
    echo "✅ Модель готова к использованию!"
    cat results/metrics.json
fi

