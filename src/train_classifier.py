import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X = np.load("data/processed/X_features.npy").astype(np.float32)
y = np.load("data/processed/y_labels.npy").astype(np.int64)

CLASS_NAMES = ["speech", "stationary", "non-stationary", "impulsive"]

# Normalize features (each band scaled to similar range — helps training stability,
# same reason we log-compressed energies earlier)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)

# Class weights: inverse frequency, so rare classes (non-stationary) get more
# "attention" from the loss function during training
class_counts = np.bincount(y_train)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(class_counts)

class NoiseClassifier(nn.Module):
    def __init__(self, in_dim=16, hidden=32, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)   # no softmax here — CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        return self.net(x)

model = NoiseClassifier()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(150):
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = loss_fn(logits, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 15 == 0:
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean()
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Accuracy: {val_acc.item():.3f}")

# Final evaluation
with torch.no_grad():
    val_preds = model(X_val_t).argmax(dim=1).numpy()

print("\n--- Confusion Matrix (rows=true, cols=predicted) ---")
print(CLASS_NAMES)
print(confusion_matrix(y_val, val_preds))

print("\n--- Classification Report ---")
print(classification_report(y_val, val_preds, target_names=CLASS_NAMES))

torch.save(model.state_dict(), "data/processed/noise_classifier.pt")
print("\nModel saved to data/processed/noise_classifier.pt")