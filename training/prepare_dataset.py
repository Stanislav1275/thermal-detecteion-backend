"""
Подготовка FLIR ADAS датасета для обучения YOLO модели.
Конвертирует аннотации из COCO JSON формата в YOLO формат.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def convert_coco_to_yolo(
    json_path: str,
    images_dir: str,
    output_dir: str,
    class_map: dict = None,
    min_size: int = 5,
    min_ratio: float = 0.01,
    max_aspect_ratio: float = 10.0
):
    """
    Конвертирует COCO аннотации в YOLO формат.
    
    Процесс конвертации:
    1. Загрузка COCO JSON файла с аннотациями
    2. Создание маппингов image_id -> информация об изображении
    3. Создание маппинга category_id -> название класса
    4. Обработка каждой аннотации:
       - Фильтрация по классам (только person и car)
       - Фильтрация маленьких и вытянутых объектов
       - Конвертация координат из абсолютных (COCO) в нормализованные (YOLO)
    5. Сохранение меток в формате YOLO (class_id x_center y_center width height)
    6. Копирование изображений в выходную директорию
    
    Формат YOLO:
    - Координаты нормализованы относительно размера изображения (0.0 - 1.0)
    - Формат: class_id x_center y_center width height
    - Один файл .txt на изображение
    
    Args:
        json_path: Путь к COCO JSON файлу с аннотациями
        images_dir: Директория с изображениями
        output_dir: Выходная директория для YOLO датасета
        class_map: Маппинг классов {'class_name': class_id}
        min_size: Минимальный размер объекта в пикселях
        min_ratio: Минимальный размер объекта относительно изображения
        max_aspect_ratio: Максимальное соотношение сторон объекта
    """
    if class_map is None:
        class_map = {'person': 0, 'car': 1}
    
    # Создание структуры директорий для YOLO датасета
    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Этап 1: Загрузка COCO JSON файла
    print(f"Загрузка аннотаций из {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Этап 2: Создание маппингов для быстрого доступа
    # Маппинг image_id -> информация об изображении (имя файла, размеры)
    image_id_to_info = {}
    for img in data['images']:
        image_id_to_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Маппинг category_id -> название класса
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Этап 3: Обработка аннотаций и конвертация в YOLO формат
    labels = defaultdict(list)
    skipped_small = 0
    skipped_aspect = 0
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]
        cat_name = category_id_to_name[cat_id]
        
        # Пропускаем классы, которых нет в class_map
        if cat_name not in class_map:
            continue
        
        x, y, w, h = bbox
        info = image_id_to_info[img_id]
        img_w, img_h = info['width'], info['height']
        
        # Фильтрация маленьких объектов
        if w < min_size or h < min_size or w / img_w < min_ratio or h / img_h < min_ratio:
            skipped_small += 1
            continue
        
        # Фильтрация вытянутых объектов
        if w / h > max_aspect_ratio or h / w > max_aspect_ratio:
            skipped_aspect += 1
            continue
        
        # Этап 4: Конвертация координат из COCO в YOLO формат
        # COCO: [x, y, width, height] - абсолютные координаты, x,y - левый верхний угол
        # YOLO: [x_center, y_center, width, height] - нормализованные координаты (0.0-1.0)
        x_center = (x + w/2) / img_w  # Центр по X, нормализованный
        y_center = (y + h/2) / img_h  # Центр по Y, нормализованный
        w_norm = w / img_w  # Ширина, нормализованная
        h_norm = h / img_h  # Высота, нормализованная
        
        labels[img_id].append(
            f"{class_map[cat_name]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )
    
    # Этап 5: Сохранение меток и копирование изображений
    # Для каждого изображения создается файл .txt с метками в формате YOLO
    copied_images = 0
    for img_id, info in image_id_to_info.items():
        file_name = os.path.basename(info['file_name'])
        txt_file_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_file_name)
        img_src_path = os.path.join(images_dir, info['file_name'])
        img_dst_path = os.path.join(images_output_dir, file_name)
        
        # Запись меток
        with open(txt_path, 'w') as f:
            if img_id in labels:
                f.write("\n".join(labels[img_id]) + "\n")
            else:
                # Пустой файл для изображений без объектов нужных классов
                f.write("")
        
        # Копирование изображений
        if os.path.exists(img_src_path):
            shutil.copy(img_src_path, img_dst_path)
            copied_images += 1
        else:
            print(f"⚠ Изображение не найдено: {img_src_path}")
    
    print(f"\n✅ Конвертация завершена:")
    print(f"   - Обработано изображений: {len(image_id_to_info)}")
    print(f"   - Скопировано изображений: {copied_images}")
    print(f"   - Создано меток: {len(os.listdir(labels_dir))}")
    print(f"   - Пропущено маленьких объектов: {skipped_small}")
    print(f"   - Пропущено вытянутых объектов: {skipped_aspect}")
    
    return output_dir


def prepare_flir_dataset(
    dataset_root: str,
    output_root: str,
    splits: list = None,
    class_map: dict = None
):
    """
    Подготавливает полный FLIR ADAS датасет для обучения.
    
    Args:
        dataset_root: Корневая директория датасета FLIR ADAS
        output_root: Выходная директория для YOLO датасета
        splits: Список сплитов для обработки ['train', 'val']
        class_map: Маппинг классов
    """
    if splits is None:
        splits = ['train', 'val']
    
    if class_map is None:
        class_map = {'person': 0, 'car': 1}
    
    print(f"Подготовка FLIR ADAS датасета...")
    print(f"Классы: {list(class_map.keys())}")
    
    for split in splits:
        print(f"\n📂 Обработка {split.upper()} split...")
        
        json_path = os.path.join(
            dataset_root, 
            f"FLIR_ADAS_1_3/{split}/thermal_annotations.json"
        )
        # Путь к изображениям: в JSON file_name содержит "thermal_8_bit/FLIR_XXXXX.jpeg"
        # Изображения находятся в FLIR_ADAS_1_3/{split}/thermal_8_bit/
        images_dir = os.path.join(
            dataset_root,
            f"FLIR_ADAS_1_3/{split}"
        )
        output_dir = os.path.join(output_root, split)
        
        if not os.path.exists(json_path):
            print(f"⚠ JSON файл не найден: {json_path}")
            continue
        
        if not os.path.exists(images_dir):
            print(f"⚠ Директория с изображениями не найдена: {images_dir}")
            continue
        
        convert_coco_to_yolo(
            json_path=json_path,
            images_dir=images_dir,
            output_dir=output_dir,
            class_map=class_map
        )
    
    print(f"\n✅ Подготовка датасета завершена!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Подготовка FLIR ADAS датасета для YOLO')
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Корневая директория FLIR ADAS датасета'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='./datasets/yolo',
        help='Выходная директория для YOLO датасета'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Сплиты для обработки'
    )
    
    args = parser.parse_args()
    
    if args.dataset_root is None:
        print("❌ Укажите --dataset-root с путем к FLIR ADAS датасету")
        print("Пример: python prepare_dataset.py --dataset-root /path/to/flir")
        exit(1)
    
    prepare_flir_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        splits=args.splits
    )

