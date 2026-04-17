# Graph Social Networks Analysis

Выпускная квалификационная работа на тему:  
**«Исследование графов больших социальных сетей методами машинного обучения»**

## Установка

```bash
git clone https://github.com/yungBaSe/graph-social-analysis.git
cd graph-social-analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Структура проекта

- `src/data/` — загрузка реальных графов с SNAP.
- `src/features/` — вычисление структурных метрик.
- `src/embeddings/` — обучение эмбеддингов (Node2Vec, GraphSAGE).
- `src/generation/` — генерация синтетических графов (ER, WS, BA, BTER).
- `src/visualization/` — визуализация графов, метрик и эмбеддингов.
- `src/sampling/` — методы сэмплирования больших графов.
- `notebooks/` — исследовательские ноутбуки.

# Данные и артефакты (игнорируются Git'ом)

- `data/raw/` — скачанные архивы с SNAP
- `data/processed/` — кешированные графы, эмбеддинги, метрики
- `data/extracted/` — распакованные файлы из ZIP-архивов
- `models/` — обученные модели (.pt, .model, .pkl)
- `figures/` — сохранённые визуализации

## Запуск

# Интерактивный режим

```bash
jupyter notebook notebooks/
```

# Пакетный запуск всех ноутбуков 

```bash
export MPLBACKEND=Agg
jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
```