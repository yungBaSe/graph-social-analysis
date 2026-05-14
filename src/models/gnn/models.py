# src/models/gnn/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GATv2Conv, TransformerConv, GINConv, JumpingKnowledge, SGConv
)

from ..base import BaseModel
from ..registry import register_model

@register_model("gcn")
class GCN(BaseModel, nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5, **kwargs):
        BaseModel.__init__(self, name="gcn", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeddings:
            return x
        x = self.convs[-1](x, edge_index)
        return x


@register_model("graphsage")
class GraphSAGE(BaseModel, nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5, **kwargs):
        BaseModel.__init__(self, name="graphsage", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeddings:
            return x
        x = self.convs[-1](x, edge_index)
        return x


@register_model("gat")
class GAT(BaseModel, nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.6, **kwargs):
        BaseModel.__init__(self, name="gat", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.convs.append(GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout))

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeddings:
            return x
        x = self.convs[-1](x, edge_index)
        return x


@register_model("gatv2")
class GATv2(BaseModel, nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.6, **kwargs):
        BaseModel.__init__(self, name="gatv2", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(num_features, hidden_dim, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.convs.append(GATv2Conv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout))

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeddings:
            return x
        x = self.convs[-1](x, edge_index)
        return x


@register_model("graph_transformer")
class GraphTransformer(BaseModel, nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.5, **kwargs):
        BaseModel.__init__(self, name="graph_transformer", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.convs.append(TransformerConv(hidden_dim, num_classes, heads=1, dropout=dropout))

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeddings:
            return x
        x = self.convs[-1](x, edge_index)
        return x

@register_model("gin")
class GIN(BaseModel, nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5, **kwargs):
        BaseModel.__init__(self, name="gin", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_features if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else num_classes
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeddings:
            return x
        x = self.convs[-1](x, edge_index)
        return x

@register_model("jknet")
class JKNet(BaseModel, nn.Module):
    """
    Jumping Knowledge Network, объединяющая выходы всех слоёв.
    Параметр conv_type задаёт тип свёртки: 'gcn', 'gat', 'gatv2', 'sage', 'gin'.
    mode: 'cat' (конкатенация), 'max', 'lstm'.
    """
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, conv_type: str = "gcn", mode: str = "cat",
                 heads: int = 4, dropout: float = 0.5, **kwargs):
        BaseModel.__init__(self, name="jknet", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mode = mode
        self.dropout = dropout
        self.convs = nn.ModuleList()

        conv_map = {
            'gcn': GCNConv,
            'gat': GATConv,
            'gatv2': GATv2Conv,
            'sage': SAGEConv,
            'gin': GINConv
        }
        ConvClass = conv_map[conv_type.lower()]

        for i in range(num_layers):
            in_dim = num_features if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim  # на выходе JK скрытый размер
            if conv_type in ('gat', 'gatv2'):
                if i < num_layers - 1:
                    self.convs.append(ConvClass(in_dim, out_dim, heads=heads, dropout=dropout, concat=True))
                    # скрытый размер увеличивается в heads раз
                    if i == 0:
                        hidden_dim = out_dim * heads
                else:
                    self.convs.append(ConvClass(in_dim, out_dim, heads=1, concat=False))
            elif conv_type == 'gin':
                mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
                self.convs.append(ConvClass(mlp, train_eps=True))
            else:
                self.convs.append(ConvClass(in_dim, out_dim))

        # Агрегация JK
        if mode == 'cat':
            self.jk = JumpingKnowledge(mode='cat')
            final_in_dim = num_layers * hidden_dim
        elif mode == 'max':
            self.jk = JumpingKnowledge(mode='max')
            final_in_dim = hidden_dim
        elif mode == 'lstm':
            self.jk = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=num_layers)
            final_in_dim = hidden_dim
        else:
            raise ValueError(f"Unknown JK mode: {mode}")

        self.final_linear = nn.Linear(final_in_dim, num_classes)

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.jk(xs)
        if return_embeddings:
            return x
        x = self.final_linear(x)
        return x
    
@register_model("sgc")
class SGC(BaseModel, nn.Module):
    def __init__(self, num_features, num_classes, K=2, **kwargs):
        BaseModel.__init__(self, name="sgc", **kwargs)
        nn.Module.__init__(self)
        self.conv = SGConv(num_features, num_classes, K=K, cached=True)

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        return self.conv(x, edge_index)
    
class GCNIIConv(nn.Module):
        """Слой GCNII с начальным остатком и отождествлением (без нормализации)."""
        def __init__(self, channels, alpha=0.1, theta=0.5):
            super().__init__()
            self.alpha = alpha
            self.theta = theta
            self.weight = nn.Parameter(torch.empty(channels, channels))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)

        def forward(self, x, x_0, edge_index):
            from torch_geometric.utils import add_self_loops, degree
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            out = x.new_zeros(x.size())
            out.index_add_(0, row, norm.view(-1, 1) * x[col])
            out = torch.matmul(out, self.weight)
            out = (1 - self.alpha) * out + self.alpha * x_0
            out = (1 - self.theta) * out + self.theta * torch.matmul(x, self.weight)
            return out

@register_model("gcnii")
class GCNII(BaseModel, nn.Module):
    """Graph Convolutional Network with Initial Residual and Identity Mapping."""

    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=2,
                 alpha=0.1, lmbda=0.5, dropout=0.5, **kwargs):
        BaseModel.__init__(self, name="gcnii", **kwargs)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.lmbda = lmbda
        self.dropout = dropout

        # Входной слой
        self.fc_in = nn.Linear(num_features, hidden_dim)
        # Скрытые слои
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNIIConv(hidden_dim, alpha=alpha, theta=lmbda))
        # Выходной слой
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc_out.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x_0 = F.relu(self.fc_in(x))
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_out(x)
        return x