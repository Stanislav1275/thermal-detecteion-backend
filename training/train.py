
import os
from ultralytics import YOLO
import torch


def train_thermal_model(
    data_yaml: str = None,
    model_size: str = None,
    epochs: int = None,
    imgsz: int = None,
    batch: int = None,
    device: str = None,
    project: str = "training/runs",
    name: str = "thermal_detection",
    **kwargs
):
    if data_yaml is None:
        data_yaml = os.getenv("TRAINING_DATA_YAML", "thermal.yaml")
    if model_size is None:
        model_size = os.getenv("TRAINING_DEFAULT_MODEL_SIZE", "n")
    if epochs is None:
        epochs = int(os.getenv("TRAINING_DEFAULT_EPOCHS", "100"))
    if imgsz is None:
        imgsz = int(os.getenv("TRAINING_DEFAULT_IMGSZ", "416"))
    if batch is None:
        batch = int(os.getenv("TRAINING_DEFAULT_BATCH", "16"))
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"\n📱 Устройство: {device.upper()}")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Файл конфигурации не найден: {data_yaml}")
    
    model_name = f"yolov8{model_size}.pt"
    print(f"📦 Загрузка модели: {model_name}")
    model = YOLO(model_name)
    
    workers = 0 if device == "mps" else 8
    
    train_params = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": project,
        "name": name,
        "patience": 10,
        "save": True,
        "val": True,
        "plots": True,
        "verbose": True,
        "amp": True,
        "fliplr": 0.5,
        "hsv_v": 0.2,
        "mosaic": 0.5,
        "mixup": 0.05,
    }
    train_params.update(kwargs)
    
    print(f"\n⚙️  Конфигурация:")
    print(f"   Модель: YOLOv8{model_size}")
    print(f"   Batch: {batch}")
    print(f"   Размер изображений: {imgsz}x{imgsz}")
    print(f"   Эпохи: {epochs}")
    print(f"   AMP: {train_params['amp']}\n")
    
    print("🚀 Начинаем обучение...\n")
    results = model.train(**train_params)
    
    best_model_path = os.path.join(project, name, "weights", "best.pt")
    
    if not os.path.exists(best_model_path):
        runs_dir = os.path.join("training", "runs") if os.path.exists("training") else "runs"
        if os.path.exists(runs_dir):
            import glob
            pattern = os.path.join(runs_dir, "thermal_detection*", "weights", "best.pt")
            matches = glob.glob(pattern)
            if matches:
                matches.sort(key=os.path.getmtime, reverse=True)
                best_model_path = matches[0]
                print(f"📦 Найдена модель в последнем запуске: {best_model_path}")
    
    if os.path.exists(best_model_path):
        target_path = "models/best.pt"
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        import shutil
        shutil.copy(best_model_path, target_path)
        file_size = os.path.getsize(target_path) / (1024 * 1024)
        print(f"\n✅ Лучшая модель сохранена: {target_path} ({file_size:.2f} MB)")
    else:
        print(f"\n⚠️  Модель не найдена: {best_model_path}")
    
    print(f"\n✅ Обучение завершено!")
    print(f"📁 Результаты: {os.path.join(project, name)}")
    
    return results, model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение YOLOv8 на термальных данных')
    parser.add_argument(
        '--data',
        type=str,
        default='thermal.yaml',
        help='Путь к YAML файлу конфигурации датасета'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Размер модели (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Количество эпох обучения'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=416,
        help='Размер изображений (416 оптимален для M4)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Размер батча'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Устройство (cpu, mps, cuda) или None для автоопределения'
    )
    
    args = parser.parse_args()
    
    train_thermal_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

