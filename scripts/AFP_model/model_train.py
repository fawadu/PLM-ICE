#!/usr/bin/env python3
"""
This script trains a neural network classifier on protein sequence embeddings
and evaluates its performance with various metrics and visualizations.
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
                             matthews_corrcoef, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset


# Path configurations
POSITIVE_EMBEDDINGS_PATH = "PATH/TO/POSITIVE.faa"
NEGATIVE_EMBEDDINGS_PATH = "PATH/TO/NEGATIVE.faa"
MODEL_SAVE_PATH = "PATH/TO/OUTPUT.txt"

# Training hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 25
BATCH_SIZE = 32

# Random seed for reproducibility
SEED = 42

# ==================================


class ProteinClassifier(nn.Module):
    """Neural network classifier for protein sequences."""

    def __init__(self, input_size: int):
        """
        Initialize the neural network.

        Args:
            input_size: Size of the input features
        """
        super(ProteinClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = x.view(-1, self.fc1.in_features)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_embeddings(file_path: str) -> List[torch.Tensor]:
    """Load embeddings from pickle file."""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data["embeddings"]


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
    plt.show()


def plot_tsne_embeddings(X: np.ndarray, y: np.ndarray) -> None:
    """Visualize embeddings using t-SNE."""
    tsne = TSNE(n_components=2, random_state=SEED)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                color='blue', label='RefNeg', alpha=0.6, edgecolor='k')
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                color='orange', label='AFPs', alpha=0.6, edgecolor='k')

    plt.title('t-SNE Visualization of Training Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()


def main():
    # Set random seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load and prepare data
    print("Loading data...")
    positive_embeddings = load_embeddings(POSITIVE_EMBEDDINGS_PATH)
    negative_embeddings = load_embeddings(NEGATIVE_EMBEDDINGS_PATH)

    # Create labels (1 for positive, 0 for negative)
    positive_labels = [1] * len(positive_embeddings)
    negative_labels = [0] * len(negative_embeddings)

    # Combine and shuffle datasets
    combined = list(zip(positive_embeddings + negative_embeddings,
                        positive_labels + negative_labels))
    random.shuffle(combined)
    X, y = zip(*combined)

    X = torch.stack(X)
    y = torch.tensor(y)

    # Visualize data with t-SNE
    plot_tsne_embeddings(X.numpy(), y.numpy())

    # Split into train and test sets
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

    print("\nStarting training...")
    print(f"Training Parameters: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}\n")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze(1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted_train = torch.sigmoid(outputs) > 0.5
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        all_predictions = []

        with torch.no_grad():
            for sequences, labels in test_loader:
                outputs = model(sequences).squeeze(1)
                predicted = torch.sigmoid(outputs) > 0.5
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels.float()).item()
                all_predictions.extend(predicted.cpu().numpy())

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

    # Plot training curves
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)

    # Calculate and print final metrics
    precision = precision_score(y_test, all_predictions, average='macro')
    recall = recall_score(y_test, all_predictions, average='macro')
    f1 = f1_score(y_test, all_predictions, average='macro')
    mcc = matthews_corrcoef(y_test, all_predictions)
    auc_roc = roc_auc_score(y_test, all_predictions)

    # Calculate specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, all_predictions).ravel()
    specificity = tn / (tn + fp)

    print("\nFinal Test Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Specificity: {specificity:.4f}")


if __name__ == "__main__":
    main()