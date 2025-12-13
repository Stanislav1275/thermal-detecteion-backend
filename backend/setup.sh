#!/bin/bash
# Скрипт для настройки conda окружения

set -e

echo "🔧 Настройка окружения для Thermal Detection System..."

# Проверка conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda не найдена в PATH."
    echo ""
    echo "Варианты решения:"
    echo "1. Инициализируйте conda: ~/miniconda3/bin/conda init zsh && source ~/.zshrc"
    echo "2. Используйте venv: ./setup_venv.sh"
    echo "3. См. SETUP_CONDА.md для подробных инструкций"
    echo ""
    exit 1
fi

# Создание окружения
echo "📦 Создание conda окружения 'thermal-detection'..."
conda create -n thermal-detection python=3.11 -y

# Активация окружения
echo "✅ Активация окружения..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thermal-detection

# Установка зависимостей
echo "📥 Установка зависимостей из environment.yml..."
conda env update -f environment.yml --prune

echo ""
echo "✅ Окружение настроено!"
echo ""
echo "Для активации окружения выполните:"
echo "  conda activate thermal-detection"
echo ""
echo "Для проверки установки:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"MPS available: {torch.backends.mps.is_available()}\")'"

