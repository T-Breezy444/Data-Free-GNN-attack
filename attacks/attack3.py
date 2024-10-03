import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TypeIIIAttack:
    def __init__(self, generator, surrogate_model1, surrogate_model2, victim_model, device, 
                 noise_dim, num_nodes, feature_dim,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        self.generator = generator
        self.surrogate_model1 = surrogate_model1
        self.surrogate_model2 = surrogate_model2
        self.victim_model = victim_model
        self.device = device
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=generator_lr)
        self.surrogate_optimizer1 = optim.Adam(self.surrogate_model1.parameters(), lr=surrogate_lr)
        self.surrogate_optimizer2 = optim.Adam(self.surrogate_model2.parameters(), lr=surrogate_lr)
        
        self.criterion = nn.CrossEntropyLoss()
        self.n_generator_steps = n_generator_steps
        self.n_surrogate_steps = n_surrogate_steps

    def generate_graph(self):
        z = torch.randn(1, self.noise_dim).to(self.device)
        features, adj = self.generator(z)
        edge_index = self.generator.adj_to_edge_index(adj)
        return features, edge_index

    def train_generator(self):
        self.generator.train()
        self.surrogate_model1.eval()
        self.surrogate_model2.eval()

        total_loss = 0
        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            surrogate_output1 = self.surrogate_model1(features, edge_index)
            surrogate_output2 = self.surrogate_model2(features, edge_index)

            # Compute disagreement loss
            loss = -torch.mean(torch.std(torch.stack([surrogate_output1, surrogate_output2]), dim=0))
            loss.backward()

            self.generator_optimizer.step()
            total_loss += loss.item()

        return total_loss / self.n_generator_steps

    def train_surrogate(self):
        self.generator.eval()
        self.surrogate_model1.train()
        self.surrogate_model2.train()

        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer1.zero_grad()
            self.surrogate_optimizer2.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output1 = self.surrogate_model1(features, edge_index)
            surrogate_output2 = self.surrogate_model2(features, edge_index)

            loss1 = self.criterion(surrogate_output1, victim_output.argmax(dim=1))
            loss2 = self.criterion(surrogate_output2, victim_output.argmax(dim=1))
            
            # Combine losses and backpropagate once
            combined_loss = loss1 + loss2
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.surrogate_model1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.surrogate_model2.parameters(), max_norm=1.0)
            
            self.surrogate_optimizer1.step()
            self.surrogate_optimizer2.step()

            total_loss += combined_loss.item() / 2

        return total_loss / self.n_surrogate_steps

    def attack(self, num_queries, log_interval=10):
        generator_losses = []
        surrogate_losses = []

        pbar = tqdm(range(num_queries), desc="Attacking")
        for query in pbar:
            gen_loss = self.train_generator()
            surr_loss = self.train_surrogate()

            generator_losses.append(gen_loss)
            surrogate_losses.append(surr_loss)

            if (query + 1) % log_interval == 0:
                pbar.set_postfix({
                    'Gen Loss': f"{gen_loss:.4f}",
                    'Surr Loss': f"{surr_loss:.4f}"
                })

        return (self.surrogate_model1, self.surrogate_model2), generator_losses, surrogate_losses

def run_attack(generator, surrogate_model1, surrogate_model2, victim_model, num_queries, device, 
               noise_dim, num_nodes, feature_dim):
    attack = TypeIIIAttack(generator, surrogate_model1, surrogate_model2, victim_model, device, 
                           noise_dim, num_nodes, feature_dim)
    return attack.attack(num_queries)
