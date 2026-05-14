# src/evaluation/experiments.py
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.data_loader import get_dataset
from src.embeddings.graph_embeddings import compute_dataset_embeddings
from src.features.preprocessing import preprocess_node_features
from src.models import (
    get_model,
    make_node_classification_data,
    make_link_prediction_data,
    NodeClassificationTrainer,
    LinkPredictionTrainer,
)
from src.sampling.graph_sampling import sample_dataset


def _get_experiment_cache_path(
    task: str,
    dataset: str,
    model: str,
    use_features: bool,
    use_embeddings: bool,
    emb_method: str,
    **kwargs,
) -> Path:
    """Генерирует путь к кэшу эксперимента."""
    parts = [task, dataset, model]
    if use_features:
        parts.append("feat")
    if use_embeddings:
        parts.append(f"emb_{emb_method}")
    for k, v in sorted(kwargs.items()):
        if k in ["dimensions", "seed", "n_runs"]:
            parts.append(f"{k}={v}")
    key = "_".join(parts)
    cache_dir = Path(__file__).resolve().parent.parent.parent / "results"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{key}.pkl"


def run_node_classification_experiment(
    dataset_name: str,
    model: str = "gcn",
    use_features: bool = True,
    use_embeddings: bool = False,
    emb_method: str = "node2vec",
    dimensions: int = 128,
    seed: int = 42,
    n_runs: int = 5,
    force_recompute: bool = False,
    balance_loss: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    cache_path = _get_experiment_cache_path(
        "node_clf", dataset_name, model, use_features, use_embeddings, emb_method,
        dimensions=dimensions, seed=seed, n_runs=n_runs, balance_loss=balance_loss, **kwargs
    )
    if not force_recompute and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"▶ Node CLF: {dataset_name} | {model} | features={use_features} | emb={use_embeddings} | balance={balance_loss} | runs={n_runs}")

    dataset = get_dataset(dataset_name, verbose=False)
    G = dataset["graph"]
    labels_df = dataset.get("labels")
    if labels_df is None:
        raise ValueError(f"Датасет {dataset_name} не содержит меток для node classification.")
    masks = dataset.get("masks")
    masks_np = {k: masks[k] for k in ['train_mask', 'val_mask', 'test_mask'] if k in masks} if masks else None

    nodes = sorted(G.nodes())
    target_col = dataset.get("target_column") or labels_df.columns[0]
    y = labels_df.reindex(nodes)[target_col].values
    if y is None or len(np.unique(y)) < 2:
        raise ValueError("Недостаточно классов для классификации.")

    X = None
    if use_features:
        raw = dataset.get("features")
        if raw is not None:
            if hasattr(raw, "reindex"):
                raw = raw.reindex(nodes).fillna(0)
            X = preprocess_node_features(raw, verbose=False)

    if use_embeddings:
        emb = compute_dataset_embeddings(dataset_name, method=emb_method, dimensions=dimensions, seed=seed)
        X_emb = np.array([emb[n] for n in nodes], dtype=np.float32)
        X = np.hstack([X, X_emb]) if X is not None else X_emb

    if X is None:
        raise ValueError("Нужно указать use_features=True или use_embeddings=True")
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)

    accuracies, macro_f1s, micro_f1s, weighted_f1s = [], [], [], []
    for run in range(n_runs):
        current_seed = seed + run
        if masks_np is not None and run == 0:
            data_dict = make_node_classification_data(G, y, node_features=X, masks=masks_np)
        else:
            data_dict = make_node_classification_data(G, y, node_features=X, random_state=current_seed)

        data = data_dict["data"]
        model_obj = get_model(
            model,
            num_features=data.x.shape[1],
            num_classes=data_dict["n_classes"],
            hidden_dim=dimensions,
            seed=current_seed,
            **kwargs,
        )
        trainer = NodeClassificationTrainer(model_obj, seed=current_seed)
        trainer.fit(data, epochs=200, early_stopping=20, verbose=False, balance=balance_loss)
        eval_metrics = trainer.evaluate(data, mask=data.test_mask)

        accuracies.append(eval_metrics["accuracy"])
        macro_f1s.append(eval_metrics["f1_macro"])
        micro_f1s.append(eval_metrics.get("f1_micro", float("nan")))
        weighted_f1s.append(eval_metrics.get("f1_weighted", float("nan")))

    result = {
        "dataset": dataset_name,
        "model": model,
        "use_features": use_features,
        "use_embeddings": use_embeddings,
        "emb_method": emb_method if use_embeddings else None,
        "dimensions": dimensions,
        "n_runs": n_runs,
        "base_seed": seed,
        "balance_loss": balance_loss,
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "macro_f1_mean": np.mean(macro_f1s),
        "macro_f1_std": np.std(macro_f1s),
        "micro_f1_mean": np.mean(micro_f1s),
        "micro_f1_std": np.std(micro_f1s),
        "weighted_f1_mean": np.mean(weighted_f1s),
        "weighted_f1_std": np.std(weighted_f1s),
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"✅ Node CLF finished: Acc={result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    return result


def run_link_prediction_experiment(
    dataset_name: str,
    method: str = "graphsage",
    use_features: bool = False,
    use_embeddings: bool = False,
    emb_method: str = "node2vec",
    dimensions: int = 128,
    seed: int = 42,
    n_runs: int = 5,
    sample_ratio: Optional[float] = None,
    force_recompute: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Запуск link prediction с многократными запусками."""
    cache_path = _get_experiment_cache_path(
        "link_pred", dataset_name, method, use_features, use_embeddings, emb_method,
        dimensions=dimensions, seed=seed, n_runs=n_runs, sample_ratio=sample_ratio, **kwargs
    )
    if not force_recompute and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"▶ Link Pred: {dataset_name} | {method} | features={use_features} | emb={use_embeddings} | runs={n_runs}")

    dataset = get_dataset(dataset_name, verbose=False)
    G = dataset["graph"]

    # Признаки (если запрошены)
    node_features = None
    if use_features:
        raw = dataset.get("features")
        if raw is not None:
            nodes = sorted(G.nodes())
            if hasattr(raw, "reindex"):
                raw = raw.reindex(nodes).fillna(0)
            node_features = preprocess_node_features(raw, verbose=False)
            node_features = np.asarray(node_features, dtype=np.float32)
            node_features = np.nan_to_num(node_features, nan=0.0)

    # Эмбеддинги (если запрошены или если нет фич)
    if use_embeddings or node_features is None:
        if sample_ratio is not None:
            # Для семплов кэшируем под именем с семплом
            emb_name = f"{dataset_name}_sampled_{sample_ratio:.3f}"
        else:
            emb_name = dataset_name
        emb = compute_dataset_embeddings(emb_name, method=emb_method, dimensions=dimensions, seed=seed)
        nodes = sorted(G.nodes())
        X_emb = np.array([emb[n] for n in nodes], dtype=np.float32)
        if node_features is not None:
            node_features = np.hstack([node_features, X_emb])
        else:
            node_features = X_emb

    if node_features is None:
        raise ValueError("Нужно указать use_features=True или use_embeddings=True")
    node_features = np.nan_to_num(node_features, nan=0.0)

    aucs, aps, hits_50 = [], [], []
    for run in range(n_runs):
        current_seed = seed + run
        data_dict = make_link_prediction_data(
            G, node_features=node_features, random_state=current_seed
        )
        data = data_dict["data"]

        if method in ["logistic", "mlp"]:
            # Используем sklearn-модели на Hadamard-признаках
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import roc_auc_score, average_precision_score

            def hadamard(edge_index):
                src, dst = edge_index[0], edge_index[1]
                return node_features[src] * node_features[dst]

            X_train = np.vstack([hadamard(data_dict["train_pos"]), hadamard(data_dict["train_neg"])])
            y_train = np.hstack([np.ones(data_dict["train_pos"].size(1)), np.zeros(data_dict["train_neg"].size(1))])
            if method == "logistic":
                clf = LogisticRegression(max_iter=1000, random_state=current_seed, n_jobs=-1)
            else:
                clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=current_seed, early_stopping=True)
            clf.fit(X_train, y_train)

            X_test = np.vstack([hadamard(data_dict["test_pos"]), hadamard(data_dict["test_neg"])])
            y_test = np.hstack([np.ones(data_dict["test_pos"].size(1)), np.zeros(data_dict["test_neg"].size(1))])
            y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_score))
            aps.append(average_precision_score(y_test, y_score))
            hits_50.append(float("nan"))  # эвристики Hits пока не считаем
        else:
            model_obj = get_model(method, num_features=data.x.shape[1], num_classes=2, hidden_dim=dimensions, **kwargs)
            trainer = LinkPredictionTrainer(model_obj, seed=current_seed)
            trainer.fit(data, epochs=80, early_stopping=10, verbose=False)
            eval_metrics = trainer.evaluate(data, stage="test")
            aucs.append(eval_metrics["auc"])
            aps.append(eval_metrics["ap"])
            hits_50.append(eval_metrics.get("hits_50", float("nan")))

    result = {
        "dataset": dataset_name,
        "method": method,
        "use_features": use_features,
        "use_embeddings": use_embeddings,
        "emb_method": emb_method if use_embeddings else None,
        "dimensions": dimensions,
        "n_runs": n_runs,
        "base_seed": seed,
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "ap_mean": np.mean(aps),
        "ap_std": np.std(aps),
        "hits_50_mean": np.mean(hits_50) if not np.isnan(np.mean(hits_50)) else None,
        "hits_50_std": np.std(hits_50) if not np.isnan(np.std(hits_50)) else None,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"✅ Link Pred finished: AUC={result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
    return result


def run_full_experiment(
    dataset_name: str,
    task: str,
    method: str = "node2vec",
    dimensions: int = 128,
    seed: int = 42,
    force_recompute: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    # Оставлено для совместимости, но не используется в новом пайплайне
    raise NotImplementedError("Use run_node_classification_experiment or run_link_prediction_experiment directly.")