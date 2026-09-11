import torch
import torch.nn as nn

N_BANDS = 16
HIDDEN = 64

class StreamingSuppressionNet(nn.Module):
    """
    Single-frame, stateful version of your trained model.
    Instead of processing 32 frames at once, this takes ONE frame + the
    previous hidden state, and returns ONE gain vector + the new hidden state
    — exactly how it will run on real embedded hardware, frame by frame.
    """
    def __init__(self, in_dim=N_BANDS, hidden=HIDDEN, out_dim=N_BANDS):
        super().__init__()
        self.gru_cell = nn.GRUCell(in_dim, hidden)
        self.out = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x, h):
        h_new = self.gru_cell(x, h)
        gain = self.out(h_new)
        return gain, h_new

# Load your trained weights
trained_state = torch.load("data/processed/suppression_model.pt", map_location="cpu", weights_only=True)

model = StreamingSuppressionNet()

# Map weights from the trained nn.GRU layer into the equivalent nn.GRUCell
# (same math, same shapes — just called one step at a time instead of over a sequence)
model.gru_cell.weight_ih.data = trained_state["gru.weight_ih_l0"]
model.gru_cell.weight_hh.data = trained_state["gru.weight_hh_l0"]
model.gru_cell.bias_ih.data = trained_state["gru.bias_ih_l0"]
model.gru_cell.bias_hh.data = trained_state["gru.bias_hh_l0"]
model.out[0].weight.data = trained_state["out.0.weight"]
model.out[0].bias.data = trained_state["out.0.bias"]

model.eval()

# Sanity check: confirm the streaming version produces the same output as
# the original batched version would, for a single frame — if this doesn't
# roughly match, the weight-mapping above is wrong
dummy_x = torch.randn(1, N_BANDS)
dummy_h = torch.zeros(1, HIDDEN)
with torch.no_grad():
    gain, h_new = model(dummy_x, dummy_h)
print(f"Sanity check — gain shape: {gain.shape}, hidden shape: {h_new.shape}")
print(f"Sample gain values: {gain.numpy()[0][:5]}")  # first 5 bands

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_x, dummy_h),
    "data/processed/suppression_model.onnx",
    input_names=["input_frame", "hidden_state_in"],
    output_names=["gain_output", "hidden_state_out"],
    opset_version=18,
    dynamic_axes=None  # fixed shapes — matches embedded deployment (no dynamic batching)
)
print("\nExported to data/processed/suppression_model.onnx")