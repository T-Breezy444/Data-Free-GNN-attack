import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SurrogateModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)
