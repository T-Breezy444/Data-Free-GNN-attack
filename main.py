import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0].to(device)

# Define a smaller GNN model
class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Initialize victim model
victim_model = SimpleGNN(dataset.num_features, 32, dataset.num_classes).to(device)

# Train victim model
def train_victim_model(model, data, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(F.log_softmax(out[data.train_mask], dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

print("Training victim model...")
train_victim_model(victim_model, data)

# Define smaller generator model
class GraphGenerator(nn.Module):
    def __init__(self, noise_dim, num_nodes, feature_dim):
        super(GraphGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(noise_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_nodes * feature_dim)
        self.fc4 = nn.Linear(128, num_nodes * num_nodes)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        features = torch.tanh(self.fc3(h)).view(self.num_nodes, self.feature_dim)
        adj = torch.sigmoid(self.fc4(h)).view(self.num_nodes, self.num_nodes)
        adj = (adj + adj.t()) / 2
        adj = adj * (1 - torch.eye(self.num_nodes, device=adj.device))
        return features, adj

# Initialize generator and surrogate model
noise_dim = 32
num_nodes = 500  # Reduce number of nodes
feature_dim = dataset.num_features
generator = GraphGenerator(noise_dim, num_nodes, feature_dim).to(device)
surrogate_model = SimpleGNN(dataset.num_features, 32, dataset.num_classes).to(device)

# Define attack function (Type I attack as per the paper)
def type_i_attack(generator, surrogate_model, victim_model, num_queries, device):
    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    surrogate_optimizer = optim.Adam(surrogate_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    generator_losses = []
    surrogate_losses = []

    for query in tqdm(range(num_queries)):
        # Train generator
        for _ in range(2):  # n_G = 2 as per the paper
            generator_optimizer.zero_grad()
            z = torch.randn(1, noise_dim).to(device)
            features, adj = generator(z)
            
            edge_index = adj.nonzero().t()
            
            with torch.no_grad():
                victim_output = victim_model(features, edge_index)
            surrogate_output = surrogate_model(features, edge_index)

            loss = -criterion(surrogate_output, victim_output.argmax(dim=1))

            # Zeroth-order optimization
            epsilon = 1e-4
            u = torch.randn_like(z)
            perturbed_z = z + epsilon * u
            perturbed_features, perturbed_adj = generator(perturbed_z)
            
            perturbed_edge_index = perturbed_adj.nonzero().t()
            
            with torch.no_grad():
                perturbed_victim_output = victim_model(perturbed_features, perturbed_edge_index)
            perturbed_surrogate_output = surrogate_model(perturbed_features, perturbed_edge_index)
            perturbed_loss = -criterion(perturbed_surrogate_output, perturbed_victim_output.argmax(dim=1))

            estimated_gradient = (perturbed_loss - loss) / epsilon * u
            z.grad = estimated_gradient

            generator_optimizer.step()
            generator_losses.append(loss.item())

        # Train surrogate model
        for _ in range(5):  # n_S = 5 as per the paper
            surrogate_optimizer.zero_grad()
            z = torch.randn(1, noise_dim).to(device)
            features, adj = generator(z)
            
            edge_index = adj.nonzero().t()

            with torch.no_grad():
                victim_output = victim_model(features, edge_index)
            surrogate_output = surrogate_model(features, edge_index)

            loss = criterion(surrogate_output, victim_output.argmax(dim=1))
            loss.backward()
            surrogate_optimizer.step()
            surrogate_losses.append(loss.item())

        if query % 10 == 0:
            print(f"Query {query}: Gen Loss = {generator_losses[-1]:.4f}, Surr Loss = {surrogate_losses[-1]:.4f}")

        # Clear cache to free up memory
        torch.cuda.empty_cache()

    return surrogate_model, generator_losses, surrogate_losses

# Run attack
print("Running attack...")
num_queries = 400  # Reduced number of queries
trained_surrogate, generator_losses, surrogate_losses = type_i_attack(generator, surrogate_model, victim_model, num_queries, device)

# Evaluate models
def evaluate_models(victim_model, surrogate_model, data):
    victim_model.eval()
    surrogate_model.eval()
    
    with torch.no_grad():
        victim_out = victim_model(data.x, data.edge_index)
        surrogate_out = surrogate_model(data.x, data.edge_index)
        
        victim_preds = victim_out.argmax(dim=1)
        surrogate_preds = surrogate_out.argmax(dim=1)

    accuracy = accuracy_score(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu())
    f1 = f1_score(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu(), average='weighted')
    conf_matrix = confusion_matrix(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu(), labels=range(dataset.num_classes))

    return accuracy, f1, conf_matrix

print("Evaluating models...")
accuracy, f1, conf_matrix = evaluate_models(victim_model, trained_surrogate, data)

print(f"Accuracy of surrogate model: {accuracy:.4f}")
print(f"F1 Score of surrogate model: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(dataset.num_classes)
plt.xticks(tick_marks, range(dataset.num_classes), rotation=45)
plt.yticks(tick_marks, range(dataset.num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss')
plt.plot(surrogate_losses, label='Surrogate Loss')
plt.title('Losses over time')
plt.xlabel('Query')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses_over_time.png')
plt.close()

print("Attack completed. Results saved in 'confusion_matrix.png' and 'losses_over_time.png'")

# Calculate improvement over random guessing
random_accuracy = 1 / dataset.num_classes
random_f1 = f1_score([0] * len(data.y[data.test_mask]), data.y[data.test_mask].cpu(), average='weighted')

accuracy_improvement = (accuracy - random_accuracy) / random_accuracy * 100
f1_improvement = (f1 - random_f1) / random_f1 * 100

print(f"Random guessing accuracy: {random_accuracy:.4f}")
print(f"Random guessing F1 Score: {random_f1:.4f}")
print(f"Accuracy improvement over random guessing: {accuracy_improvement:.2f}%")
print(f"F1 Score improvement over random guessing: {f1_improvement:.2f}%")
