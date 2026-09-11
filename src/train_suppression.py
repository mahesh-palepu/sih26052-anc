import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

X = np.load("data/processed/X_suppression.npy").astype(np.float32)
Y = np.load("data/processed/Y_suppression.npy").astype(np.float32)

SEQ_LEN = 32   # how many consecutive frames the GRU sees at once (~320ms of context
               # at our 10ms hop) — this becomes your algorithmic latency budget later

def make_sequences(X, Y, seq_len):
    n_seqs = len(X) // seq_len
    X = X[:n_seqs * seq_len].reshape(n_seqs, seq_len, X.shape[1])
    Y = Y[:n_seqs * seq_len].reshape(n_seqs, seq_len, Y.shape[1])
    return X, Y

X_seq, Y_seq = make_sequences(X, Y, SEQ_LEN)

# simple 90/10 split, no shuffling — audio sequences should stay in temporal order
split = int(0.9 * len(X_seq))
X_train, X_val = X_seq[:split], X_seq[split:]
Y_train, Y_val = Y_seq[:split], Y_seq[split:]

X_train_t = torch.from_numpy(X_train)
Y_train_t = torch.from_numpy(Y_train)
X_val_t = torch.from_numpy(X_val)
Y_val_t = torch.from_numpy(Y_val)

class SuppressionNet(nn.Module):
    def __init__(self, in_dim=16, hidden=64, out_dim=16):
        super().__init__()
        # hidden=24 is deliberately small — this whole model needs to fit an MCU later,
        # not chase the highest possible accuracy right now
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()   # gains must be 0-1, same reasoning as toy_nn's Sigmoid
        )

    def forward(self, x):
        h, _ = self.gru(x)           # h: (batch, seq_len, hidden) — output at every timestep
        return self.out(h)           # gain prediction at every timestep

model = SuppressionNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(500):
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, Y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, Y_val_t)
        print(f"Epoch {epoch:3d} | Train MSE: {loss.item():.5f} | Val MSE: {val_loss.item():.5f}")

torch.save(model.state_dict(), "data/processed/suppression_model.pt")
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel saved. Total parameters: {n_params}")