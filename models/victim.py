import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VictimModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(VictimModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.25, training=self.training)  # Paper: p=0.5
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

def create_victim_model_cora():
    input_dim = 1433
    hidden_dim = 64  # Paper: 128
    output_dim = 7
    return VictimModel(input_dim, hidden_dim, output_dim)

def create_victim_model_computers():
    input_dim = 767
    hidden_dim = 64  # Paper: 128
    output_dim = 10
    return VictimModel(input_dim, hidden_dim, output_dim)

def create_victim_model_pubmed():
    input_dim = 500
    hidden_dim = 64  # Paper: 128
    output_dim = 3
    return VictimModel(input_dim, hidden_dim, output_dim)

def create_victim_model_ogb_arxiv():
    input_dim = 128
    hidden_dim = 128  # Paper: 256
    output_dim = 40
    num_layers = 2  # Paper: 3
    return VictimModel(input_dim, hidden_dim, output_dim, num_layers)
