# src/evaluation/community.py
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering, KMeans
import community as community_louvain
import torch
from torch_geometric.nn import DMoNPooling
from torch_geometric.utils import to_undirected, to_dense_adj, to_dense_batch
from torch_geometric.data import Data, Batch

def overlapping_to_hard_labels(circles: Dict[int, Dict[str, List[int]]]) -> Dict[int, int]:
    """
    Преобразует пересекающиеся круги в жёсткую разметку (метод большинства).
    circles: {ego_id: {circle_name: [user_ids]}}
    Возвращает: {node_id: label_id}
    """
    node_to_circles = defaultdict(list)
    for ego, circle_dict in circles.items():
        for circle_name, members in circle_dict.items():
            for member in members:
                node_to_circles[member].append(circle_name)
    labels = {}
    for node, circle_list in node_to_circles.items():
        # Выбираем самый частый круг
        most_common = Counter(circle_list).most_common(1)[0][0]
        labels[node] = most_common
    # Преобразуем строковые имена в целочисленные метки
    unique_names = sorted(set(labels.values()))
    name_to_int = {name: i for i, name in enumerate(unique_names)}
    return {node: name_to_int[name] for node, name in labels.items()}


def evaluate_communities(true_labels: Dict[int, int], pred_partition: Dict[int, int]) -> Dict[str, float]:
    """
    Вычисляет NMI и ARI между эталонной разметкой и предсказанной.
    """
    # Убедимся, что порядок узлов одинаковый
    nodes = sorted(true_labels.keys())
    y_true = [true_labels[n] for n in nodes]
    y_pred = [pred_partition.get(n, -1) for n in nodes]
    # Если есть -1 (узел не назначен), то исключим его из метрик
    mask = [i for i, v in enumerate(y_pred) if v != -1]
    y_true = [y_true[i] for i in mask]
    y_pred = [y_pred[i] for i in mask]
    return {
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred)
    }


def run_louvain(G: nx.Graph) -> Dict[int, int]:
    G_u = G.to_undirected() if G.is_directed() else G
    partition = community_louvain.best_partition(G_u)
    return partition


def run_leiden(G: nx.Graph, seed: int = 42, resolution: float = 0.01) -> Dict[int, int]:
    """Leiden-алгоритм с CPMVertexPartition."""
    try:
        import leidenalg
        import igraph as ig

        G_u = G.to_undirected() if G.is_directed() else G
        nodes = list(G_u.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_u.edges()]
        g = ig.Graph(edges=edges, directed=False)

        partition = leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            resolution_parameter=resolution,
            n_iterations=-1,
            seed=seed,
            max_comm_size=0
        )
        return {nodes[i]: partition.membership[i] for i in range(len(nodes))}
    except ImportError:
        print("leidenalg не установлен. Использую Louvain.")
        return run_louvain(G)


def run_spectral(G: nx.Graph, n_clusters: int, random_state: int = 42) -> Dict[int, int]:
    G_u = G.to_undirected() if G.is_directed() else G
    adj = nx.to_scipy_sparse_array(G_u)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=random_state, assign_labels='discretize')
    labels = clustering.fit_predict(adj)
    return {node: labels[i] for i, node in enumerate(G_u.nodes())}


def run_kmeans_emb(embeddings: Dict[int, np.ndarray], n_clusters: int, random_state: int = 42) -> Dict[int, int]:
    nodes = sorted(embeddings.keys())
    X = np.array([embeddings[n] for n in nodes])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    return {node: kmeans.labels_[i] for i, node in enumerate(nodes)}

def run_dmon(G: nx.Graph, dimensions: int = 128, epochs: int = 200,
             lr: float = 1e-3, seed: int = 42, n_clusters: Optional[int] = None) -> Dict[int, int]:
    import torch
    from torch_geometric.nn import DMoNPooling
    from torch_geometric.utils import to_undirected, to_dense_adj, to_dense_batch
    from torch_geometric.data import Data, Batch

    # 1. Фиксируем seed для воспроизводимости
    torch.manual_seed(seed)

    # 2. Подготовка графа
    G_u = G.to_undirected() if G.is_directed() else G
    nodes = list(G_u.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edge_list = [[node_to_idx[u], node_to_idx[v]] for u, v in G_u.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index)

    # Признаки: используем переданные, либо единичную матрицу
    if hasattr(G_u, 'features') and G_u.features is not None:
        x = torch.tensor(G_u.features, dtype=torch.float32)
    else:
        x = torch.eye(len(nodes), dtype=torch.float32)

    # 3. Создаём батч из одного графа
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = batch.to(device)

    # 4. Преобразуем в плотный формат: получаем признаки [1, N, F] и маску [1, N]
    x_dense, mask = to_dense_batch(batch.x, batch.batch)
    adj = to_dense_adj(batch.edge_index, batch.batch)  # [1, N, N]

    # 5. Параметры кластеризации
    n_nodes = len(nodes)
    if n_clusters is None:
        n_clusters = max(2, int(len(nodes) ** 0.5))
    model = DMoNPooling(x_dense.size(-1), n_clusters).to(device)
    model.reset_parameters()  # сброс параметров
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6. Обучение
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = model(x_dense, adj, mask)
        # Используем ТОЛЬКО modularity loss и cluster loss (как в статье и официальном примере)
        loss = spectral_loss + cluster_loss
        loss.backward()
        optimizer.step()

    # 7. Получение финальных меток
    model.eval()
    with torch.no_grad():
        s, _, _, _, _, _ = model(x_dense, adj, mask)
        # s shape: [1, N, C] -> [N, C]
        assignments = s.argmax(dim=-1).squeeze(0).cpu().numpy()

    return {node: int(assignments[i]) for i, node in enumerate(nodes)}

def run_nocd(G: nx.Graph, dimensions: int = 128, epochs: int = 200,
             lr: float = 1e-3, seed: int = 42, n_clusters: Optional[int] = None
             ) -> Dict[int, np.ndarray]:
    """
    Neural Overlapping Community Detection (NOCD).
    Возвращает словарь node_id -> np.array размера C (вероятности принадлежности к сообществам).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    torch.manual_seed(seed)
    np.random.seed(seed)

    G_u = G.to_undirected() if G.is_directed() else G
    nodes = list(G_u.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    edge_list = [[node_to_idx[u], node_to_idx[v]] for u, v in G_u.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index)

    # Признаки: используем фичи графа, иначе единичные
    if hasattr(G_u, 'features') and G_u.features is not None:
        x = torch.tensor(G_u.features, dtype=torch.float32)
    else:
        x = torch.eye(len(nodes), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    n_nodes = len(nodes)
    if n_clusters is None:
        n_clusters = max(2, int(n_nodes ** 0.5))

    # Энкодер: GCN
    class Encoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    encoder = Encoder(data.x.size(1), dimensions, n_clusters).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    encoder.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = encoder(data.x, data.edge_index)          # [N, C]
        # Реконструкция матрицы смежности через inner product и сигмоиду
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))   # [N, N]
        # Потеря: binary cross entropy на всех парах (используем true adjacency)
        adj_true = to_dense_adj(data.edge_index, max_num_nodes=n_nodes)[0]
        loss = F.binary_cross_entropy(adj_pred, adj_true, reduction='mean')
        loss.backward()
        optimizer.step()

    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index).cpu().numpy()

    # Возвращаем словарь с вероятностями
    return {node: z[i] for i, node in enumerate(nodes)}

def run_hdbscan(
    G: nx.Graph,
    embeddings: Dict[int, np.ndarray],
    min_cluster_size: int = 5,
    metric: str = 'euclidean',
    seed: int = 42
) -> Dict[int, int]:
    """
    HDBSCAN-кластеризация на эмбеддингах вершин.
    Возвращает словарь {node_id: cluster_id} без шумовых вершин (label = -1).
    """
    from sklearn.cluster import HDBSCAN

    nodes = sorted(G.nodes())
    X = np.array([embeddings[n] for n in nodes])
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = hdb.fit_predict(X)

    partition = {}
    for i, node in enumerate(nodes):
        if labels[i] != -1:
            partition[node] = int(labels[i])
    return partition