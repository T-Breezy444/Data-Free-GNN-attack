import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TypeIAttack:
    def __init__(self, generator, surrogate_model, victim_model, device, 
                 noise_dim, num_nodes, feature_dim,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        self.generator = generator
        self.surrogate_model = surrogate_model
        self.victim_model = victim_model
        self.device = device
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=generator_lr)
        self.surrogate_optimizer = optim.Adam(self.surrogate_model.parameters(), lr=surrogate_lr)
        self.criterion = nn.CrossEntropyLoss()

        self.n_generator_steps = n_generator_steps
        self.n_surrogate_steps = n_surrogate_steps

    def generate_graph(self):
        z = torch.randn(1, self.noise_dim).to(self.device)
        features, adj = self.generator(z)
        edge_index = adj.nonzero().t()
        return features, edge_index

    def train_generator(self):
        self.generator.train()
        self.surrogate_model.eval()

        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)

            loss = -self.criterion(surrogate_output, victim_output.argmax(dim=1))

            # Zeroth-order optimization
            epsilon = 1e-6
            z = torch.randn(1, self.noise_dim).to(self.device)
            u = torch.randn_like(z)
            perturbed_z = z + epsilon * u
            perturbed_features, perturbed_adj = self.generator(perturbed_z)
            perturbed_edge_index = perturbed_adj.nonzero().t()

            with torch.no_grad():
                perturbed_victim_output = self.victim_model(perturbed_features, perturbed_edge_index)
            perturbed_surrogate_output = self.surrogate_model(perturbed_features, perturbed_edge_index)
            perturbed_loss = -self.criterion(perturbed_surrogate_output, perturbed_victim_output.argmax(dim=1))

            estimated_gradient = (perturbed_loss - loss) / epsilon * u
            z.grad = estimated_gradient

            self.generator_optimizer.step()

        return loss.item()

    def train_surrogate(self):
        self.generator.eval()
        self.surrogate_model.train()

        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)

            loss = self.criterion(surrogate_output, victim_output.argmax(dim=1))
            loss.backward()
            self.surrogate_optimizer.step()

            total_loss += loss.item()

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

        return self.surrogate_model, generator_losses, surrogate_losses

def run_attack(generator, surrogate_model, victim_model, num_queries, device, 
               noise_dim, num_nodes, feature_dim):
    attack = TypeIAttack(generator, surrogate_model, victim_model, device, 
                         noise_dim, num_nodes, feature_dim)
    return attack.attack(num_queries)
