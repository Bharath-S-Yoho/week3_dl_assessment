import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

torch.manual_seed(1)


class CsvDataset(Dataset):
    """Dataset wrapper with the required __len__ and __getitem__ methods."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.layers(x)


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Fit scaler only on training data to avoid data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(CsvDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(CsvDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(CsvDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = MLP(X_train.shape[1])
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Params:", param_count)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 6
    no_improvement = 0
    train_losses = []

    for epoch in range(1, 41):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                predictions = model(features).argmax(dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
        val_acc = accuracy_score(all_labels, all_predictions)
        print(f"Epoch {epoch:02d}: train loss={train_loss:.4f}, validation accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "model_state_dict.pth")

    # Reload weights into a new model and compare validation accuracy.
    reloaded = MLP(X_train.shape[1])
    reloaded.load_state_dict(torch.load("model_state_dict.pth", map_location="cpu"))
    reloaded.eval()
    model.eval()

    def get_accuracy(net, loader):
        predictions, labels_all = [], []
        with torch.no_grad():
            for features, labels in loader:
                predictions.extend(net(features).argmax(dim=1).tolist())
                labels_all.extend(labels.tolist())
        return accuracy_score(labels_all, predictions)

    original_acc = get_accuracy(model, val_loader)
    reloaded_acc = get_accuracy(reloaded, val_loader)
    test_acc = get_accuracy(reloaded, test_loader)
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Reloaded model accuracy: {reloaded_acc:.4f} (original {original_acc:.4f})")
    print(f"Test accuracy: {test_acc:.4f}")

    baseline = LogisticRegression(max_iter=2000)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"Scikit-learn Logistic Regression test accuracy: {baseline_acc:.4f}")


if __name__ == "__main__":
    main()
