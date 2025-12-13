#!/bin/bash
# Скрипт для настройки окружения через venv (без conda)

set -e

echo "🔧 Настройка окружения через venv для Thermal Detection System..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d. -f1,2)
echo "📦 Найден Python: $PYTHON_VERSION"

# Проверка версии Python (нужен 3.11+)
if [ "$(printf '%s\n' "3.11" "$PYTHON_MAJOR_MINOR" | sort -V | head -n1)" != "3.11" ]; then
    echo "⚠️  Предупреждение: Рекомендуется Python 3.11+, найден $PYTHON_VERSION"
    read -p "Продолжить? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Создание виртуального окружения
echo "📦 Создание виртуального окружения..."
python3 -m venv venv

# Активация окружения
echo "✅ Активация окружения..."
source venv/bin/activate

# Обновление pip
echo "📥 Обновление pip..."
pip install --upgrade pip

# Установка зависимостей
echo "📥 Установка зависимостей из requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Окружение настроено!"
echo ""
echo "Для активации окружения выполните:"
echo "  source venv/bin/activate"
echo ""
echo "Для проверки установки:"
echo "  python check_environment.py"

