import torch
import torch.nn as nn
import numpy as np

# Generate toy data: points in 2D, label = 1 if inside unit circle, else 0
np.random.seed(0)
X = np.random.uniform(-1.5, 1.5, (2000, 2)).astype(np.float32)
y = (np.sum(X**2, axis=1) < 1.0).astype(np.float32).reshape(-1, 1)

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

# A tiny neural net: 2 inputs -> 16 hidden -> 16 hidden -> 1 output
class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()   # squashes output to 0-1, so it reads as a probability
        )

    def forward(self, x):
        return self.net(x)

model = ToyNet()
loss_fn = nn.BCELoss()                                 # binary cross-entropy: standard loss for yes/no problems
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()          # clear old gradients
    pred = model(X_t)              # forward pass
    loss = loss_fn(pred, y_t)      # how wrong are we?
    loss.backward()                # backprop: compute gradients
    optimizer.step()               # nudge weights to reduce loss

    if epoch % 20 == 0:
        acc = ((pred > 0.5) == y_t).float().mean()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {acc.item():.3f}")

print("\nFinal check — model should predict close to 1 for (0,0), close to 0 for (1.4,1.4):")
test_pts = torch.tensor([[0.0, 0.0], [1.4, 1.4]])
print(model(test_pts).detach().numpy())