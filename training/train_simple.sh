#!/bin/bash
# Простой запуск обучения

set -e

EPOCHS=${1:-100}  # По умолчанию 100 эпох, можно указать другое: ./train_simple.sh 50

echo "🚀 Запуск обучения модели..."
echo "   Эпохи: $EPOCHS"
echo ""

python train.py \
    --data thermal.yaml \
    --model n \
    --epochs $EPOCHS \
    --batch 16 \
    --imgsz 640

echo ""
echo "✅ Обучение завершено!"
echo "   Модель сохранена в: training/models/best.pt"

