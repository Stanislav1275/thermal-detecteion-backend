"""
Скрипт для проверки окружения и зависимостей.
"""

import sys
import importlib


def check_package(package_name, import_name=None):
    """Проверяет наличие пакета."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def check_environment():
    """Проверяет окружение и зависимости."""
    print("🔍 Проверка окружения...\n")
    
    # Проверка Python версии
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major != 3 or python_version.minor < 11:
        print("⚠️  Рекомендуется Python 3.11+")
    print()
    
    # Список обязательных пакетов
    required_packages = {
        'torch': ('torch', 'torch'),
        'torchvision': ('torchvision', 'torchvision'),
        'ultralytics': ('ultralytics', 'ultralytics'),
        'fastapi': ('fastapi', 'fastapi'),
        'uvicorn': ('uvicorn', 'uvicorn'),
        'pillow': ('pillow', 'PIL'),
        'opencv-python': ('opencv-python', 'cv2'),
        'numpy': ('numpy', 'numpy'),
        'pydantic': ('pydantic', 'pydantic'),
    }
    
    all_ok = True
    
    print("Проверка зависимостей:")
    print("-" * 50)
    for package_name, (_, import_name) in required_packages.items():
        installed, version = check_package(package_name, import_name)
        if installed:
            print(f"✅ {package_name:20s} {version}")
        else:
            print(f"❌ {package_name:20s} не установлен")
            all_ok = False
    print()
    
    # Проверка PyTorch и MPS
    try:
        import torch
        print(f"PyTorch версия: {torch.__version__}")
        
        # Проверка MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            print(f"MPS (Apple Silicon) доступен: {'✅ Да' if mps_available else '❌ Нет'}")
        else:
            print("MPS не поддерживается в этой версии PyTorch")
        print()
    except ImportError:
        print("❌ PyTorch не установлен\n")
        all_ok = False
    
    # Проверка модели
    import os
    model_path = "training/models/best.pt"
    if os.path.exists(model_path):
        print(f"✅ Модель найдена: {model_path}")
    else:
        print(f"⚠️  Модель не найдена: {model_path}")
        print("   Модель должна быть обучена перед использованием API")
    print()
    
    if all_ok:
        print("✅ Все зависимости установлены!")
        return True
    else:
        print("❌ Некоторые зависимости отсутствуют.")
        print("   Выполните: conda env update -f environment.yml --prune")
        return False


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)

