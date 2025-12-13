#!/bin/bash
set -e

EPOCHS=${1:-100}

if command -v conda &> /dev/null && conda info --envs | grep -q "thermal-detection"; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate thermal-detection
elif [ -d "../backend/venv" ]; then
    source ../backend/venv/bin/activate
fi

echo "Запуск обучения модели..."
echo "Эпохи: $EPOCHS"
echo ""

python train.py \
    --data thermal.yaml \
    --model n \
    --epochs $EPOCHS \
    --batch 16 \
    --imgsz 416

echo ""
echo "✅ Обучение завершено! Модель сохранена в: models/best.pt"

