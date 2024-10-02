import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from models.victim import create_victim_model_cora
from models.generator import GraphGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0].to(device)

# Initialize and train victim model
victim_model = create_victim_model_cora().to(device)

def train_victim_model(model, data, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

print("Training victim model...")
train_victim_model(victim_model, data)

# Initialize generator and surrogate model
noise_dim = 32
num_nodes = 500
feature_dim = dataset.num_features
generator = GraphGenerator(noise_dim, num_nodes, feature_dim, generator_type='cosine').to(device)
surrogate_model = create_victim_model_cora().to(device)

# Define attack function (Type I attack)
def type_i_attack(generator, surrogate_model, victim_model, num_queries, device):
    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    surrogate_optimizer = optim.Adam(surrogate_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    generator_losses = []
    surrogate_losses = []

    for query in tqdm(range(num_queries)):
        # Train generator
        for _ in range(2):
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
        for _ in range(5):
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

    return surrogate_model, generator_losses, surrogate_losses

# Run attack
print("Running attack...")
num_queries = 400
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
    fidelity = accuracy_score(victim_preds.cpu(), surrogate_preds.cpu())
    f1 = f1_score(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu(), average='weighted')
    conf_matrix = confusion_matrix(victim_preds[data.test_mask].cpu(), surrogate_preds[data.test_mask].cpu(), labels=range(dataset.num_classes))

    return accuracy, fidelity, f1, conf_matrix

print("Evaluating models...")
accuracy, fidelity, f1, conf_matrix = evaluate_models(victim_model, trained_surrogate, data)

# Generate PDF report
def generate_pdf_report(accuracy, fidelity, f1, conf_matrix, generator_losses, surrogate_losses):
    doc = SimpleDocTemplate("attack_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("STEALGNN Attack Report", styles['Title']))

    # Metrics table
    data = [
        ["Metric", "Value"],
        ["Accuracy", f"{accuracy:.4f}"],
        ["Fidelity", f"{fidelity:.4f}"],
        ["F1 Score", f"{f1:.4f}"]
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)

    # Confusion Matrix
    elements.append(Paragraph("Confusion Matrix", styles['Heading2']))
    conf_data = [[str(x) for x in row] for row in conf_matrix]
    conf_data.insert(0, [f"Class {i}" for i in range(len(conf_matrix))])
    t = Table(conf_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)

    doc.build(elements)

# Generate report
generate_pdf_report(accuracy, fidelity, f1, conf_matrix, generator_losses, surrogate_losses)

print(f"Accuracy of surrogate model: {accuracy:.4f}")
print(f"Fidelity of surrogate model: {fidelity:.4f}")
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

print("Attack completed. Results saved in 'confusion_matrix.png', 'losses_over_time.png', and 'attack_report.pdf'")

# Calculate improvement over random guessing
random_accuracy = 1 / dataset.num_classes
random_f1 = f1_score([0] * len(data.y[data.test_mask]), data.y[data.test_mask].cpu(), average='weighted')

accuracy_improvement = (accuracy - random_accuracy) / random_accuracy * 100
f1_improvement = (f1 - random_f1) / random_f1 * 100

print(f"Random guessing accuracy: {random_accuracy:.4f}")
print(f"Random guessing F1 Score: {random_f1:.4f}")
print(f"Accuracy improvement over random guessing: {accuracy_improvement:.2f}%")
print(f"F1 Score improvement over random guessing: {f1_improvement:.2f}%")
