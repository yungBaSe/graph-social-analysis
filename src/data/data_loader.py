# src/data/data_loader.py
"""
Модуль для загрузки графов реальных социальных сетей.
Конфигурация датасетов хранится в datasets.json.
"""

import os
import pickle
import gzip
import zipfile
import json
import shutil
import requests
import networkx as nx
from pathlib import Path

# ================================ КОНФИГУРАЦИЯ ПУТЕЙ ================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTRACTED = PROJECT_ROOT / "data" / "extracted"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_EXTRACTED.mkdir(parents=True, exist_ok=True)

# Путь к конфигу
CONFIG_PATH = Path(__file__).parent / "datasets.json"

# ================================ ЗАГРУЗКА КОНФИГА =================================
def load_datasets_config() -> dict:
    """Загружает конфигурацию датасетов из JSON-файла."""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Добавляем динамический extract_path для zip-датасетов
    for name, info in config.items():
        if info.get("type") == "zip" and "extract_subdir" in info:
            info["extract_path"] = DATA_EXTRACTED / info["extract_subdir"]
    
    return config

DATASETS = load_datasets_config()

# ================================ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===========================
def download_file(url: str, local_path: Path) -> None:
    """Скачивает файл по URL в указанный локальный путь."""
    if local_path.exists():
        print(f"✅ Файл уже существует: {local_path.name}")
        return
    print(f"🔽 Скачивание {local_path.name} из {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Сохранён: {local_path}")

def extract_zip(zip_path: Path, extract_to: Path, file_pattern: str) -> Path:
    """
    Извлекает из ZIP-архива файл, содержащий в имени file_pattern.
    Возвращает путь к извлечённому файлу.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"🗜️ Поиск файла с паттерном '{file_pattern}' в {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        
        # Ищем файл, который содержит file_pattern и заканчивается на .csv
        candidates = [f for f in all_files if file_pattern in f and f.endswith('.csv')]
        
        if not candidates:
            csv_files = [f for f in all_files if f.endswith('.csv')]
            raise FileNotFoundError(
                f"Файл с паттерном '{file_pattern}' не найден в архиве.\n"
                f"Доступные CSV-файлы: {csv_files}"
            )
        
        target_in_zip = candidates[0]
        target_filename = Path(target_in_zip).name
        extracted_file_path = extract_to / target_filename
        
        if extracted_file_path.exists():
            print(f"✅ Файл уже извлечён: {extracted_file_path}")
            return extracted_file_path
        
        print(f"📦 Извлечение {target_in_zip} -> {extracted_file_path}")
        
        temp_dir = extract_to / "_temp"
        temp_dir.mkdir(exist_ok=True)
        zip_ref.extract(target_in_zip, temp_dir)
        
        temp_file = temp_dir / target_in_zip
        temp_file.rename(extracted_file_path)
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    print(f"✅ Файл извлечён: {extracted_file_path}")
    return extracted_file_path

def load_graph_from_csv(csv_path: Path) -> nx.Graph:
    """Загружает граф из CSV-файла (формат: u, v)."""
    print(f"🧵 Чтение графа из {csv_path.name}...")
    G = nx.Graph()
    with open(csv_path, 'r') as f:
        # Пропускаем заголовок, если он есть
        first_line = f.readline().strip()
        f.seek(0)
        
        # Если первая строка не похожа на пару чисел, считаем её заголовком
        parts = first_line.split(',')
        has_header = not (len(parts) >= 2 and parts[0].strip().isdigit())
        
        for i, line in enumerate(f):
            if has_header and i == 0:
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                u, v = parts[0].strip(), parts[1].strip()
                G.add_edge(u, v)
    return G

def load_graph_from_gz(dataset_name: str) -> nx.Graph:
    """Загружает граф из сжатого .txt.gz файла."""
    info = DATASETS[dataset_name]
    gz_path = DATA_RAW / info["filename"]
    pkl_path = DATA_PROCESSED / info["processed_name"]

    if pkl_path.exists():
        print(f"📂 Загрузка готового графа из {pkl_path.name}")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    download_file(info["url"], gz_path)
    print(f"🧵 Чтение графа из {gz_path.name}...")
    G = nx.Graph()
    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = map(int, parts[:2])
                G.add_edge(u, v)
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"💾 Граф сохранён в {pkl_path.name}")
    return G

def load_graph_from_zip_csv(dataset_name: str) -> nx.Graph:
    """Загружает граф из CSV-файла, который находится внутри ZIP-архива."""
    info = DATASETS[dataset_name]
    zip_path = DATA_RAW / info["filename"]
    pkl_path = DATA_PROCESSED / info["processed_name"]

    if pkl_path.exists():
        print(f"📂 Загрузка готового графа из {pkl_path.name}")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    download_file(info["url"], zip_path)
    csv_path = extract_zip(zip_path, info["extract_path"], info["archive_name"])
    G = load_graph_from_csv(csv_path)

    with open(pkl_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"💾 Граф сохранён в {pkl_path.name}")
    return G

# ================================ ОСНОВНАЯ ФУНКЦИЯ =================================
def get_graph(name: str) -> nx.Graph:
    """Загружает граф по его имени, автоматически определяя тип обработки."""
    if name not in DATASETS:
        raise ValueError(f"Неизвестный датасет: {name}. Доступные: {list(DATASETS.keys())}")
    
    dataset_info = DATASETS[name]
    if dataset_info["type"] == "gz":
        return load_graph_from_gz(name)
    elif dataset_info["type"] == "zip":
        return load_graph_from_zip_csv(name)
    else:
        raise NotImplementedError(f"Тип датасета '{dataset_info['type']}' не поддерживается.")

def reload_datasets() -> None:
    """Перезагружает конфиг датасетов (полезно при добавлении новых)."""
    global DATASETS
    DATASETS = load_datasets_config()
    print("✅ Конфиг датасетов перезагружен.")
    list_available_datasets()

def list_available_datasets() -> None:
    """Выводит список доступных датасетов с описанием."""
    print("Доступные датасеты:")
    for key, val in DATASETS.items():
        print(f"  - {key}: {val['description']}")