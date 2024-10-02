import torch
import torch.nn as nn

class GraphGenerator(nn.Module):
    def __init__(self, noise_dim, num_nodes, feature_dim):
        super(GraphGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.feature_gen = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * feature_dim),
            nn.Tanh()
        )

        self.threshold = 0.1

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        features = torch.tanh(self.fc3(h)).view(self.num_nodes, self.feature_dim)
        adj = torch.sigmoid(self.fc4(h)).view(self.num_nodes, self.num_nodes)
        adj = (adj + adj.t()) / 2
        adj = adj * (1 - torch.eye(self.num_nodes, device=adj.device))
        return features, adj
