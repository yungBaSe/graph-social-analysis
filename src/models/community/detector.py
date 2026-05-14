# src/models/community/detector.py
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional
import community as community_louvain


class CommunityDetector:
    """
    Удобный класс для Community Detection.
    Поддерживает:
    - Классический Louvain на графе
    - Louvain на эмбеддингах (embedding-based)
    """

    def __init__(self, method: str = "louvain"):
        self.method = method.lower()

    def detect(self, 
               G: nx.Graph | nx.DiGraph, 
               embeddings: Optional[Dict[int, np.ndarray]] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Основной метод.
        Если embeddings переданы — запускает Louvain на эмбеддингах.
        Иначе — на самом графе.
        """
        G_u = G.to_undirected() if G.is_directed() else G

        if embeddings is not None:
            # Embedding-based community detection
            return self._embedding_based(G_u, embeddings, **kwargs)
        else:
            # Traditional community detection on graph
            return self._traditional(G_u, **kwargs)

    def _traditional(self, G: nx.Graph, **kwargs) -> Dict[str, Any]:
        """Классический Louvain на графе."""
        try:
            partition = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(partition, G)
        except:
            # fallback
            communities = list(nx.community.greedy_modularity_communities(G))
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(G, communities)

        return {
            "partition": partition,
            "modularity": float(modularity),
            "n_communities": len(set(partition.values())),
            "method": "louvain"
        }

    def _embedding_based(self, G: nx.Graph, embeddings: Dict[int, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Louvain на эмбеддингах узлов."""
        # Создаём similarity graph на основе косинусного сходства эмбеддингов
        from sklearn.metrics.pairwise import cosine_similarity
        nodes = list(G.nodes())
        emb_matrix = np.array([embeddings[n] for n in nodes])

        sim_matrix = cosine_similarity(emb_matrix)
        threshold = kwargs.get("threshold", 0.7)

        H = nx.Graph()
        H.add_nodes_from(nodes)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sim_matrix[i, j] > threshold:
                    H.add_edge(nodes[i], nodes[j])

        # Запускаем Louvain на similarity-графе
        return self._traditional(H)

    def get_modularity(self, G: nx.Graph, embeddings: Optional[Dict] = None, **kwargs) -> float:
        """Удобный shortcut для получения только modularity."""
        result = self.detect(G, embeddings, **kwargs)
        return result["modularity"]


# Для удобства
def detect_communities(G: nx.Graph | nx.DiGraph, embeddings: Optional[Dict] = None, **kwargs):
    detector = CommunityDetector()
    return detector.detect(G, embeddings, **kwargs)