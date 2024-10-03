import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphGenerator(nn.Module):
    def __init__(self, noise_dim, num_nodes, feature_dim, generator_type='cosine', threshold=0.1):
        super(GraphGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.generator_type = generator_type
        self.threshold = threshold
       
        # Feature generator
        self.feature_gen = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * feature_dim),
            nn.Tanh()
        )
       
        # Full parameterization structure generator
        if generator_type == 'full_param':
            self.structure_gen = nn.Sequential(
                nn.Linear(noise_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, num_nodes * num_nodes),
                nn.Sigmoid()
            )

    def forward(self, z):
        # Generate features
        features = self.feature_gen(z).view(self.num_nodes, self.feature_dim)
       
        # Generate adjacency matrix
        if self.generator_type == 'cosine':
            adj = self.cosine_similarity_generator(features)
        elif self.generator_type == 'full_param':
            adj = self.full_param_generator(z)
        else:
            raise ValueError("Invalid generator type. Choose 'cosine' or 'full_param'.")
        
        # Normalize adjacency matrix
        adj = adj / adj.sum(1, keepdim=True).clamp(min=1)
        
        return features, adj

    def cosine_similarity_generator(self, features):
        # Compute cosine similarity
        norm_features = F.normalize(features, p=2, dim=1)
        adj = torch.mm(norm_features, norm_features.t())
       
        # Apply threshold
        adj = (adj > self.threshold).float()
       
        # Remove self-loops
        adj = adj * (1 - torch.eye(self.num_nodes, device=adj.device))
       
        return adj

    def full_param_generator(self, z):
        adj = self.structure_gen(z).view(self.num_nodes, self.num_nodes)
       
        # Make symmetric
        adj = (adj + adj.t()) / 2
       
        # Remove self-loops
        adj = adj * (1 - torch.eye(self.num_nodes, device=adj.device))
       
        return adj

    def adj_to_edge_index(self, adj):
        return adj.nonzero().t()

    def self_supervised_training(self, x, adj, model):
        # Implement self-supervised denoising task
        self.train()
        
        # Add noise to features
        noise = torch.randn_like(x) * 0.1
        noisy_x = x + noise
        
        # Use the model to denoise
        edge_index = self.adj_to_edge_index(adj)
        denoised_x = model(noisy_x, edge_index)
        
        # Compute reconstruction loss
        loss = F.mse_loss(denoised_x, x)
        
        return loss

class DenoisingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, input_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
