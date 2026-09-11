import numpy as np
import torch
import torch.nn as nn

N_BANDS = 16
HIDDEN = 64

class StreamingSuppressionNet(nn.Module):
    def __init__(self, in_dim=N_BANDS, hidden=HIDDEN, out_dim=N_BANDS):
        super().__init__()
        self.gru_cell = nn.GRUCell(in_dim, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, out_dim), nn.Sigmoid())

    def forward(self, x, h):
        h_new = self.gru_cell(x, h)
        gain = self.out(h_new)
        return gain, h_new

trained_state = torch.load("data/processed/suppression_model.pt", map_location="cpu", weights_only=True)
model = StreamingSuppressionNet()
model.gru_cell.weight_ih.data = trained_state["gru.weight_ih_l0"]
model.gru_cell.weight_hh.data = trained_state["gru.weight_hh_l0"]
model.gru_cell.bias_ih.data = trained_state["gru.bias_ih_l0"]
model.gru_cell.bias_hh.data = trained_state["gru.bias_hh_l0"]
model.out[0].weight.data = trained_state["out.0.weight"]
model.out[0].bias.data = trained_state["out.0.bias"]
model.eval()

# Use REAL feature frames (not random) — pulls 5 actual frames from your dataset,
# so this test reflects real input distributions, not synthetic edge cases
X = np.load("data/processed/X_suppression.npy").astype(np.float32)
test_frames = X[100:105]  # 5 consecutive real frames

hidden = np.zeros((1, HIDDEN), dtype=np.float32)
pt_gains, pt_hiddens, inputs_used, hiddens_used = [], [], [], []

with torch.no_grad():
    for frame in test_frames:
        x_t = torch.from_numpy(frame.reshape(1, N_BANDS))
        h_t = torch.from_numpy(hidden)
        inputs_used.append(frame.copy())
        hiddens_used.append(hidden.copy().flatten())

        gain, h_new = model(x_t, h_t)
        pt_gains.append(gain.numpy().flatten())
        pt_hiddens.append(h_new.numpy().flatten())
        hidden = h_new.numpy()  # carry state forward, exactly like real streaming

np.save("data/processed/eq_check_inputs.npy", np.array(inputs_used))
np.save("data/processed/eq_check_hiddens_in.npy", np.array(hiddens_used))
np.save("data/processed/eq_check_pt_gains.npy", np.array(pt_gains))
np.save("data/processed/eq_check_pt_hiddens_out.npy", np.array(pt_hiddens))

print("Saved 5 frames of PyTorch reference output (streaming, state carried forward).")
print(f"Sample PyTorch gain (frame 0): {pt_gains[0][:5]}")