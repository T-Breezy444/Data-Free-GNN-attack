import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.5):
        super(SurrogateModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.softmax(x, dim=1)

    def train_step(self, generator, victim_model, optimizer, criterion, device):
        self.train()
        optimizer.zero_grad()
        
        z = torch.randn(1, generator.noise_dim).to(device)
        features, adj = generator(z)
        edge_index = generator.adj_to_edge_index(adj)
        
        with torch.no_grad():
            victim_output = victim_model(features, edge_index)
        surrogate_output = self(features, edge_index)
        
        loss = criterion(surrogate_output, victim_output.argmax(dim=1))
        loss.backward()
        optimizer.step()
        
        return loss.item()
