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

- `src/data/` — загрузка и обработка датасетов.
- `src/embeddings/` — обучение эмбеддингов.
- `src/evaluation/` — реализация основных экспериментов.
- `src/features/` — вычисление структурных метрик и обработка признаков.
- `src/models/` — реализации классов обучаемых моделей.
- `src/sampling/` — методы сэмплирования.
- `src/visualization/` — визуализация графов.
- `notebooks/` — исследовательские ноутбуки.

## Данные и артефакты (игнорируются Git'ом)

- `data/raw/` — скачанные архивы
- `data/pyg/` — скачанные архивы из PyTorch Geometric
- `data/processed/` — кешированные графы, эмбеддинги, метрики
- `data/extracted/` — распакованные файлы из ZIP-архивов
- `figures/` — сохранённые визуализации
- `results/` — сохранённые общие результаты 

# Запуск

Ноутбуки можно запустить вручную, также можно использовать `jupyter`.

## Установка

```bash
pip install jupyter
```

## Проверка базового функционала

```bash
jupyter notebook notebooks/demo/
```

## Запуск основных ноутбуков

```bash
jupyter notebook notebooks/
```

## Пакетный запуск всех ноутбуков 

```bash
export MPLBACKEND=Agg
jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
```

# Превью текущих итогов сравнения моделей в рамках каждой из задач

## Классификация вершин (Node Classification)

Accuracy / Macro‑F1 (среднее ± std по запускам).

| Датасет | Лучшая модель | Accuracy | Macro‑F1 |
|:---|:---|:---|:---|
| Cora | GATv2 | 0.854 ± 0.011 | 0.843 ± 0.006 |
| PubMed | GCNII | 0.886 ± 0.004 | 0.883 ± 0.004 |
| Twitch RU | GIN / GCNII | 0.753 ± 0.002 | 0.430 ± 0.001 |
| LastFM Asia | MLP | 0.745 ± 0.005 | 0.492 ± 0.008 |
| ogbn‑arxiv | GraphSAGE | 0.644 ± 0.000 | 0.408 ± 0.004 |

## Предсказание связей (Link Prediction)

AUC (среднее по запускам). Структурные эвристики приведены как базовый уровень.

| Датасет | Лучшая модель (GNN) | AUC модели | Лучшая эвристика | AUC эвристики |
|:---|:---|:---|:---|:---|
| Cora | Logistic | 0.820 | Adamic‑Adar | 0.697 |
| PubMed | GraphSAGE | 0.933 | Pref. Attach. | 0.720 |
| Facebook | GIN | 0.956 | Resource Alloc. | **0.988** |
| Twitch RU | GATv2 | 0.918 | Pref. Attach. | 0.897 |
| LastFM Asia | GAT | 0.805 | Adamic‑Adar | 0.837 |
| ogbn‑arxiv | GraphSAGE | 0.918 | Resource Alloc. | 0.895 |


## Обнаружение сообществ (Community Detection)

NMI / ARI на подграфах YouTube (100, 200, 400 ground‑truth сообществ).

| Размер | Лучший метод | NMI | ARI |
|:---|:---|:---|:---|
| 100 | HDBSCAN | 0.926 | 0.725 |
| 200 | HDBSCAN | 0.908 | 0.694 |
| 400 | HDBSCAN | 0.874 | 0.514 |

## Сэмплирование Reddit

Качество downstream‑задач на семплах (среднее по размерам 2000/4000/6000).

| Метод | NC Accuracy | LP AUC | CD NMI |
|:---|:---|:---|:---|
| Forest Fire | 0.628 | 0.534 | 0.322 |
| TIES | 0.121 | 0.628 | 0.557 |
