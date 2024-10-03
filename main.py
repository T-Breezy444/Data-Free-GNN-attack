import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# Import our custom modules
from models.victim import create_victim_model_cora, create_victim_model_computers, create_victim_model_pubmed, create_victim_model_ogb_arxiv
from models.generator import GraphGenerator
from attacks.attack1 import TypeIAttack
from attacks.attack2 import TypeIIAttack
from attacks.attack3 import TypeIIIAttack

def create_masks(num_nodes, train_ratio=0.6, val_ratio=0.2):
    indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    return train_mask, val_mask, test_mask

def main(attack_type, dataset_name):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and create victim model
    if dataset_name == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        victim_model = create_victim_model_cora().to(device)
    elif dataset_name == 'computers':
        dataset = Amazon(root='/tmp/Amazon', name='Computers', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        data.edge_index = to_undirected(data.edge_index)
        train_mask, val_mask, test_mask = create_masks(data.num_nodes)
        data.train_mask = train_mask.to(device)
        data.val_mask = val_mask.to(device)
        data.test_mask = test_mask.to(device)
        victim_model = create_victim_model_computers().to(device)
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        victim_model = create_victim_model_pubmed().to(device)
    elif dataset_name == 'ogb-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        data.train_mask = data.train_mask.to(device)
        data.val_mask = data.val_mask.to(device)
        data.test_mask = data.test_mask.to(device)
        victim_model = create_victim_model_ogb_arxiv().to(device)
    else:
        raise ValueError("Invalid dataset name. Choose 'cora', 'computers', 'pubmed', or 'ogb-arxiv'.")

    # Train victim model
    train_victim_model(victim_model, data, dataset_name)

    # Initialize generator and surrogate model(s)
    noise_dim = 32
    num_nodes = 500
    feature_dim = dataset.num_features
    output_dim = dataset.num_classes

    generator = GraphGenerator(noise_dim, num_nodes, feature_dim, generator_type='cosine').to(device)
    
    if dataset_name == 'cora':
        surrogate_model1 = create_victim_model_cora().to(device)
    elif dataset_name == 'computers':
        surrogate_model1 = create_victim_model_computers().to(device)
    elif dataset_name == 'pubmed':
        surrogate_model1 = create_victim_model_pubmed().to(device)
    elif dataset_name == 'ogb-arxiv':
        surrogate_model1 = create_victim_model_ogb_arxiv().to(device)

    # Attack parameters
    num_queries = 700
    generator_lr = 1e-6
    surrogate_lr = 0.001
    n_generator_steps = 2
    n_surrogate_steps = 5

    # Run attack based on attack_type
    print(f"Running attack type {attack_type} on {dataset_name} dataset...")

    if attack_type == 1:
        attack = TypeIAttack(generator, surrogate_model1, victim_model, device, 
                             noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                             n_generator_steps, n_surrogate_steps)
    elif attack_type == 2:
        attack = TypeIIAttack(generator, surrogate_model1, victim_model, device, 
                              noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                              n_generator_steps, n_surrogate_steps)
    elif attack_type == 3:
        if dataset_name == 'cora':
            surrogate_model2 = create_victim_model_cora().to(device)
        elif dataset_name == 'computers':
            surrogate_model2 = create_victim_model_computers().to(device)
        elif dataset_name == 'pubmed':
            surrogate_model2 = create_victim_model_pubmed().to(device)
        elif dataset_name == 'ogb-arxiv':
            surrogate_model2 = create_victim_model_ogb_arxiv().to(device)
        
        attack = TypeIIIAttack(generator, surrogate_model1, surrogate_model2, victim_model, device, 
                            noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                            n_generator_steps, n_surrogate_steps)
    else:
        raise ValueError("Invalid attack type. Choose 1, 2, or 3.")

    trained_surrogate, generator_losses, surrogate_losses = attack.attack(num_queries)

    # Evaluate models
    accuracy, fidelity, f1, conf_matrix = evaluate_models(victim_model, trained_surrogate, data)

    # Calculate random baselines
    random_accuracy, random_f1 = calculate_random_baselines(data)

    # Print and store stats
    stats = {
        "Dataset": dataset_name,
        "Attack Type": attack_type,
        "Accuracy": accuracy,
        "Fidelity": fidelity,
        "F1 Score": f1,
        "Random Accuracy": random_accuracy,
        "Random F1": random_f1,
        "Accuracy Improvement": (accuracy - random_accuracy) / random_accuracy * 100,
        "F1 Improvement": (f1 - random_f1) / random_f1 * 100
    }

    print_stats(stats)

    # Plot results
    plot_confusion_matrix(conf_matrix, output_dim, attack_type, dataset_name)
    plot_losses(generator_losses, surrogate_losses, attack_type, dataset_name)

    # Generate PDF report
    generate_pdf_report(stats, conf_matrix, attack_type, dataset_name)

def train_victim_model(model, data, dataset_name, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        if dataset_name == 'ogb-arxiv':
            loss = nn.functional.nll_loss(out[data.train_mask], data.y.squeeze()[data.train_mask])
        else:
            loss = nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                if dataset_name == 'ogb-arxiv':
                    val_loss = nn.functional.nll_loss(val_out[data.val_mask], data.y.squeeze()[data.val_mask])
                    val_acc = (val_out[data.val_mask].argmax(dim=1) == data.y.squeeze()[data.val_mask]).float().mean()
                else:
                    val_loss = nn.functional.nll_loss(val_out[data.val_mask], data.y[data.val_mask])
                    val_acc = (val_out[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).float().mean()
            model.train()
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}')

def evaluate_models(victim_model, trained_surrogate, data):
    victim_model.eval()
    if isinstance(trained_surrogate, tuple):
        surrogate_model1, surrogate_model2 = trained_surrogate
        surrogate_model1.eval()
        surrogate_model2.eval()
    else:
        surrogate_model = trained_surrogate
        surrogate_model.eval()
    
    with torch.no_grad():
        victim_out = victim_model(data.x, data.edge_index)
        if isinstance(trained_surrogate, tuple):
            surrogate_out1 = surrogate_model1(data.x, data.edge_index)
            surrogate_out2 = surrogate_model2(data.x, data.edge_index)
            surrogate_out = (surrogate_out1 + surrogate_out2) / 2  # Simple ensemble
        else:
            surrogate_out = surrogate_model(data.x, data.edge_index)
        
        victim_preds = victim_out.argmax(dim=1)
        surrogate_preds = surrogate_out.argmax(dim=1)

    accuracy = accuracy_score(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu())
    fidelity = accuracy_score(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu())
    f1 = f1_score(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu(), average='weighted')
    conf_matrix = confusion_matrix(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu())

    return accuracy, fidelity, f1, conf_matrix

def plot_confusion_matrix(conf_matrix, num_classes, attack_type, dataset_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Type {attack_type} Attack on {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_type{attack_type}_{dataset_name}.png')
    plt.close()

def plot_losses(generator_losses, surrogate_losses, attack_type, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(surrogate_losses, label='Surrogate Loss')
    plt.title(f'Losses over time - Type {attack_type} Attack on {dataset_name}')
    plt.xlabel('Query')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'losses_over_time_type{attack_type}_{dataset_name}.png')
    plt.close()

def calculate_random_baselines(data):
    num_classes = data.y.max().item() + 1
    random_preds = torch.randint(0, num_classes, data.y.shape).to(data.y.device)
    random_accuracy = accuracy_score(data.y[data.test_mask].cpu(), random_preds[data.test_mask].cpu())
    random_f1 = f1_score(data.y[data.test_mask].cpu(), random_preds[data.test_mask].cpu(), average='weighted')
    return random_accuracy, random_f1

def print_stats(stats):
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def generate_pdf_report(stats, conf_matrix, attack_type, dataset_name):
    pdf_filename = f"type{attack_type}_attack_{dataset_name}_report.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Type {attack_type} Attack on {dataset_name} Report")

    c.setFont("Helvetica", 12)
    y = height - 100
    for key, value in stats.items():
        if isinstance(value, float):
            c.drawString(50, y, f"{key}: {value:.4f}")
        else:
            c.drawString(50, y, f"{key}: {value}")
        y -= 20

    # Add confusion matrix to the report
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Confusion Matrix")
    
    table_width = 400
    table_height = 300
    x_start = (width - table_width) / 2
    y_start = height - 100 - table_height
    
    cell_width = table_width / conf_matrix.shape[1]
    cell_height = table_height / conf_matrix.shape[0]
    
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            x = x_start + j * cell_width
            y = y_start + (conf_matrix.shape[0] - 1 - i) * cell_height
            c.rect(x, y, cell_width, cell_height)
            c.setFont("Helvetica", 10)
            c.drawString(x + 2, y + 2, str(conf_matrix[i, j]))

    c.save()
    print(f"PDF report saved as {pdf_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <attack_type> <dataset_name>")
        print("attack_type: 1, 2, or 3")
        print("dataset_name: cora, computers, pubmed, or ogb-arxiv")
        sys.exit(1)
    
    try:
        attack_type = int(sys.argv[1])
        if attack_type not in [1, 2, 3]:
            raise ValueError
        dataset_name = sys.argv[2]
        if dataset_name not in ['cora', 'computers', 'pubmed', 'ogb-arxiv']:
            raise ValueError
    except ValueError:
        print("Invalid input. Please choose attack type 1, 2, or 3 and dataset name 'cora', 'computers', 'pubmed', or 'ogb-arxiv'.")
        sys.exit(1)

    main(attack_type, dataset_name)
