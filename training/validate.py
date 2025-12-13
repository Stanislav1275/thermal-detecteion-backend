"""
Валидация обученной YOLOv8 модели на тестовом наборе.
Вычисляет метрики качества и визуализирует результаты.
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def validate_model(
    model_path: str,
    data_yaml: str,
    output_dir: str = "training/results",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    visualize_samples: int = 20
):
    """
    Валидирует обученную модель на тестовом наборе.
    
    Процесс валидации:
    1. Загрузка обученной модели
    2. Запуск валидации на тестовом наборе данных
    3. Вычисление метрик качества:
       - mAP@50: средняя точность при IoU=0.5
       - mAP@50-95: средняя точность при IoU=0.5-0.95 (среднее по IoU от 0.5 до 0.95)
       - Precision: точность детекций (TP / (TP + FP))
       - Recall: полнота детекций (TP / (TP + FN))
    4. Сохранение метрик в JSON формате
    5. Генерация графиков метрик
    
    Метрики:
    - mAP@50: основная метрика для оценки качества детекции
    - Precision: доля правильных детекций среди всех детекций
    - Recall: доля найденных объектов среди всех объектов в датасете
    
    Args:
        model_path: Путь к обученной модели (.pt файл)
        data_yaml: Путь к YAML файлу конфигурации датасета
        output_dir: Директория для сохранения результатов
        conf_threshold: Порог уверенности для детекции
        iou_threshold: Порог IoU для NMS (Non-Maximum Suppression)
        visualize_samples: Количество примеров для визуализации
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Этап 1: Загрузка обученной модели
    print(f"📦 Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    print(f"\n🔍 Валидация модели на датасете: {data_yaml}")
    
    # Этап 2: Запуск валидации
    # Процесс включает:
    # - Загрузку тестовых данных из датасета
    # - Предсказания модели для каждого изображения
    # - Сравнение предсказаний с ground truth аннотациями
    # - Вычисление метрик на основе IoU (Intersection over Union)
    results = model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=True,
        save_json=True,
        save_dir=output_dir
    )
    
    # Этап 3: Извлечение и сохранение метрик
    # Метрики вычисляются Ultralytics автоматически на основе сравнения предсказаний с ground truth
    metrics = {
        "mAP50": float(results.box.map50),  # mAP при IoU=0.5
        "mAP50-95": float(results.box.map),  # mAP при IoU=0.5-0.95 (среднее)
        "precision": float(results.box.mp),  # Точность (precision)
        "recall": float(results.box.mr),  # Полнота (recall)
        "conf_threshold": conf_threshold,  # Использованный порог уверенности
        "iou_threshold": iou_threshold  # Использованный порог IoU для NMS
    }
    
    # Сохранение метрик в JSON для последующего анализа
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Метрики валидации:")
    print(f"   - mAP@50: {metrics['mAP50']:.4f}")
    print(f"   - mAP@50-95: {metrics['mAP50-95']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall: {metrics['recall']:.4f}")
    print(f"\n📊 Метрики сохранены: {metrics_path}")
    
    return results, metrics


def visualize_predictions(
    model_path: str,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    num_samples: int = 20,
    conf_threshold: float = 0.25
):
    """
    Визуализирует предсказания модели на тестовых изображениях.
    
    Процесс визуализации:
    1. Загрузка модели и тестовых изображений
    2. Получение предсказаний модели для каждого изображения
    3. Загрузка ground truth аннотаций из файлов меток
    4. Создание side-by-side визуализации:
       - Левая панель: Ground Truth (зеленые рамки)
       - Правая панель: Предсказания модели (оранжевые рамки)
    5. Сохранение визуализаций для анализа качества детекции
    
    Args:
        model_path: Путь к обученной модели
        images_dir: Директория с изображениями
        labels_dir: Директория с метками (YOLO формат)
        output_dir: Директория для сохранения визуализаций
        num_samples: Количество примеров для визуализации
        conf_threshold: Порог уверенности для фильтрации предсказаний
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    class_map = model.names
    gt_class_map = {0: 'person', 1: 'car'}
    
    # Получаем список изображений
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:num_samples]
    
    print(f"\n🎨 Визуализация {len(image_files)} примеров...")
    
    for idx, file_name in enumerate(image_files):
        img_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")
        
        # Чтение изображения
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Предсказания модели
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        preds = results[0].boxes
        
        # Создание фигуры
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Левое изображение - Ground Truth
        ax1.imshow(img_rgb)
        ax1.set_title(f"Ground Truth: {file_name}", fontsize=12)
        ax1.axis('off')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, xc, yc, bw, bh = map(float, parts)
                        x1 = (xc - bw/2) * w
                        y1 = (yc - bh/2) * h
                        width = bw * w
                        height = bh * h
                        rect = patches.Rectangle(
                            (x1, y1), width, height,
                            linewidth=2, edgecolor='green', facecolor='none'
                        )
                        ax1.add_patch(rect)
                        ax1.text(
                            x1, y1-5,
                            gt_class_map.get(int(cls_id), f"class_{int(cls_id)}"),
                            color='green', fontsize=10, weight='bold'
                        )
        
        # Правое изображение - Предсказания
        ax2.imshow(img_rgb)
        ax2.set_title(f"Predictions: {file_name}", fontsize=12)
        ax2.axis('off')
        
        for box, cls_id in zip(preds.xywh, preds.cls):
            xc, yc, bw, bh = box.tolist()
            conf = float(box.conf[0])
            x1 = (xc - bw/2)
            y1 = (yc - bh/2)
            rect = patches.Rectangle(
                (x1, y1), bw, bh,
                linewidth=2, edgecolor='orange', facecolor='none'
            )
            ax2.add_patch(rect)
            label = f"{class_map.get(int(cls_id), str(int(cls_id)))} {conf:.2f}"
            ax2.text(
                x1, y1-5, label,
                color='orange', fontsize=10, weight='bold'
            )
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"sample_{idx+1:03d}_{file_name}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 5 == 0:
            print(f"   Обработано: {idx + 1}/{len(image_files)}")
    
    print(f"\n✅ Визуализации сохранены в: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Валидация обученной YOLO модели')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к обученной модели (.pt файл)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='thermal.yaml',
        help='Путь к YAML файлу конфигурации датасета'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='training/results',
        help='Директория для сохранения результатов'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Порог уверенности'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Создать визуализации предсказаний'
    )
    
    args = parser.parse_args()
    
    results, metrics = validate_model(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        conf_threshold=args.conf
    )
    
    if args.visualize:
        # Определяем пути к тестовым данным
        data_dir = os.path.dirname(args.data)
        test_images_dir = os.path.join(data_dir, "test", "images")
        test_labels_dir = os.path.join(data_dir, "test", "labels")
        
        if os.path.exists(test_images_dir):
            visualize_predictions(
                model_path=args.model,
                images_dir=test_images_dir,
                labels_dir=test_labels_dir,
                output_dir=os.path.join(args.output, "visualizations"),
                conf_threshold=args.conf
            )

