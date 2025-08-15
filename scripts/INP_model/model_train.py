#!/usr/bin/env python3
"""
Protein Sequence Classifier Training Script

This script trains a neural network classifier on protein sequence embeddings
with early stopping and performance visualization.
"""

import pickle
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Path to positive class embeddings pickle file
POSITIVE_EMBEDDINGS_PATH = "/PATH/TO/INP.pkl"
# Path to negative class embeddings pickle file
NEGATIVE_EMBEDDINGS_PATH = "/PATH/TO/NEGATIVE.pkl"
# Path to save the trained model
MODEL_SAVE_PATH = "/PATH/TO/inp_model"

# Training hyperparameters
LEARNING_RATE = 0.00008
EPOCHS = 30
BATCH_SIZE = 4
DECISION_THRESHOLD = 0.85  # For classification

# Early stopping parameters
PATIENCE = 2
MIN_DELTA = 1e-4

# Random seed for reproducibility
SEED = 42


# ==========================================

class ProteinClassifier(nn.Module):
    """Neural network classifier for protein sequences."""

    def __init__(self, input_size: int):
        super(ProteinClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with intermediate layer outputs."""
        x = x.view(-1, self.fc1.in_features)
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3, [x1, x2]


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray,
              layer_name: str, epoch: int) -> None:
    """Visualize embeddings using t-SNE."""
    tsne = TSNE(n_components=2, random_state=SEED)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1],
                color='blue', label='Negative', alpha=0.6)
    plt.scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1],
                color='red', label='Positive', alpha=0.6)
    plt.title(f't-SNE of {layer_name} at Epoch {epoch}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.savefig(f"tsne_{layer_name.lower().replace(' ', '_')}_epoch_{epoch}.png")
    plt.close()


def load_embeddings(file_path: str) -> List[torch.Tensor]:
    """Load embeddings from pickle file."""
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data["embeddings"]


def main():
    # Set random seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load and prepare data
    print("Loading data...")
    positive_embeddings = load_embeddings(POSITIVE_EMBEDDINGS_PATH)
    negative_embeddings = load_embeddings(NEGATIVE_EMBEDDINGS_PATH)

    # Create labels and combine datasets
    positive_labels = [1] * len(positive_embeddings)
    negative_labels = [0] * len(negative_embeddings)

    combined = list(zip(positive_embeddings + negative_embeddings,
                        positive_labels + negative_labels))
    random.shuffle(combined)
    X, y = zip(*combined)

    X = torch.stack(X)
    y = torch.tensor(y)

    # Split into train and test sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = ProteinClassifier(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training tracking
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Early stopping setup
    best_loss = float('inf')
    no_improvement = 0

    print("\nStarting training...")
    print(f"Training Parameters:")
    print(f"- Learning Rate: {LEARNING_RATE}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Decision Threshold: {DECISION_THRESHOLD}\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_intermediate = []
        all_train_labels = []

        # Training loop
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs, intermediates = model(sequences)
            outputs = outputs.squeeze(1)

            # Store intermediate outputs for visualization
            all_intermediate.append([i.detach().cpu().numpy() for i in intermediates])
            all_train_labels.append(labels.cpu().numpy())

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.sigmoid(outputs) > DECISION_THRESHOLD
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Generate t-SNE plots at specific intervals
        if (epoch + 1) % 10 == 0 or epoch in [0, 2, 4, EPOCHS - 1]:
            all_train_labels = np.concatenate(all_train_labels)
            for i, layer_output in enumerate(zip(*all_intermediate)):
                layer_output = np.concatenate(layer_output)
                plot_tsne(layer_output, all_train_labels, f'Layer {i + 1}', epoch + 1)

        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        all_predictions = []

        with torch.no_grad():
            for sequences, labels in test_loader:
                outputs, _ = model(sequences)
                outputs = outputs.squeeze(1)
                predicted = torch.sigmoid(outputs) > DECISION_THRESHOLD
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels.float()).item()
                all_predictions.extend(predicted.cpu().numpy())

        # Check for early stopping
        current_loss = train_losses[-1]
        if current_loss < (best_loss - MIN_DELTA):
            best_loss = current_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Calculate validation metrics
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Calculate final metrics
    precision = precision_score(y_test, all_predictions)
    recall = recall_score(y_test, all_predictions)
    f1 = f1_score(y_test, all_predictions)
    mcc = matthews_corrcoef(y_test, all_predictions)
    auc_roc = roc_auc_score(y_test, all_predictions)

    print("\nFinal Test Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot training curves
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)


def plot_training_curves(train_losses: List[float], test_losses: List[float],
                         train_accuracies: List[float], test_accuracies: List[float]) -> None:
    """Plot training and validation curves."""
    epochs = len(train_losses)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot losses
    ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss',
             color='blue', linestyle='-')
    ax1.plot(range(1, epochs + 1), test_losses, label='Test Loss',
             color='blue', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Plot accuracies
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy',
             color='red', linestyle='-')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy',
             color='red', linestyle='--')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.title('Training and Validation Metrics')
    plt.savefig("training_metrics.png")
    plt.close()


if __name__ == "__main__":
    main()