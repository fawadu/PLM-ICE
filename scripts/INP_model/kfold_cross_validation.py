#!/usr/bin/env python3
"""
This script performs k-fold cross-validation and evaluates multiple performance metrics.
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
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset

# ========== MANUAL CONFIGURATION ==========
# Path to positive class embeddings pickle file
POSITIVE_EMBEDDINGS_PATH = "/PATH/TO/INP.pkl"
# Path to negative class embeddings pickle file
NEGATIVE_EMBEDDINGS_PATH = "/PATH/TO/NEGATIVE.pkl"

# Training hyperparameters
LEARNING_RATE = 0.00005
EPOCHS = 50
BATCH_SIZE = 4

# Cross-validation parameters
K_FOLDS = 5
TEST_SIZE = 0.2  # For initial train-test split

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


def load_embeddings(file_path: str) -> List[torch.Tensor]:
    """Load embeddings from a pickle file."""
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data["embeddings"]


def plot_metrics(train_losses: List[float], test_losses: List[float],
                 train_accuracies: List[float], test_accuracies: List[float],
                 epochs: int, fold: int) -> None:
    """Plot training and validation metrics."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot losses
    ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue', linestyle='-')
    ax1.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='blue', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='center left')

    # Plot accuracies
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='red', linestyle='-')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='red', linestyle='--')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='center right')

    plt.title(f'Fold {fold} Training Metrics (LR={LEARNING_RATE}, Batch={BATCH_SIZE})')
    plt.savefig(f"training_metrics_fold_{fold}.png")
    plt.close()


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
    combined_data = list(zip(positive_embeddings + negative_embeddings,
                           positive_labels + negative_labels))
    random.shuffle(combined_data)
    X, y = zip(*combined_data)

    X = torch.stack(X)
    y = torch.tensor(y)

    # Initial train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=SEED, test_size=TEST_SIZE, shuffle=True)

    # Initialize lists to store metrics across folds
    metrics = {
        'train_accuracy': [],
        'test_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mcc': [],
        'auc_roc': []
    }

    # Cross-validation setup
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    print(f"\nStarting {K_FOLDS}-fold cross-validation")
    print(f"Parameters: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}\n")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_train)):
        print(f'Fold {fold + 1}/{K_FOLDS}')

        # Split data for current fold
        x_train, x_test = X_train[train_idx], X_train[test_idx]
        y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

        # Create data loaders
        train_dataset = TensorDataset(x_train, y_train_fold)
        test_dataset = TensorDataset(x_test, y_test_fold)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model, loss, and optimizer
        model = ProteinClassifier(X_train.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training and validation tracking
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for sequences, labels in train_loader:
                optimizer.zero_grad()
                outputs, _ = model(sequences)
                outputs = outputs.squeeze(1)
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
            all_labels = []

            with torch.no_grad():
                for sequences, labels in test_loader:
                    outputs, _ = model(sequences)
                    outputs = outputs.squeeze(1)
                    test_loss += criterion(outputs, labels.float()).item()

                    predicted_test = torch.sigmoid(outputs) > 0.5
                    total_test += labels.size(0)
                    correct_test += (predicted_test == labels).sum().item()

                    all_predictions.extend(predicted_test.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate validation metrics
            test_loss = test_loss / len(test_loader)
            test_accuracy = 100 * correct_test / total_test
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch + 1}/{EPOCHS}, "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # Calculate final metrics for this fold
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        mcc = matthews_corrcoef(all_labels, all_predictions)
        auc_roc = roc_auc_score(all_labels, all_predictions)

        # Store metrics
        metrics['train_accuracy'].append(train_accuracies[-1])
        metrics['test_accuracy'].append(test_accuracies[-1])
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['mcc'].append(mcc)
        metrics['auc_roc'].append(auc_roc)

        print(f"\nFold {fold + 1} Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}\n")

        # Plot training/validation curves
        plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies,
                     EPOCHS, fold + 1)

    # Print summary of all folds
    print("\nFinal Metrics Across All Folds:")
    for metric, values in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: Mean={np.mean(values):.4f}, Max={np.max(values):.4f}")


if __name__ == "__main__":
    main()