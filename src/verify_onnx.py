import numpy as np
import onnx
import onnxruntime as ort

MODEL_PATH = "data/processed/suppression_model.onnx"

# 1. Check that the ONNX model structure is valid
onnx_model = onnx.load(MODEL_PATH)
onnx.checker.check_model(onnx_model)

print("ONNX structural validation: PASSED")

# 2. Show model metadata and expected tensor shapes
print("\nModel inputs:")
for item in onnx_model.graph.input:
    dims = [
        dim.dim_value if dim.dim_value > 0 else dim.dim_param
        for dim in item.type.tensor_type.shape.dim
    ]
    print(f"  {item.name}: {dims}")

print("\nModel outputs:")
for item in onnx_model.graph.output:
    dims = [
        dim.dim_value if dim.dim_value > 0 else dim.dim_param
        for dim in item.type.tensor_type.shape.dim
    ]
    print(f"  {item.name}: {dims}")

# 3. Run one streaming frame through ONNX Runtime
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_frame = np.random.randn(1, 16).astype(np.float32)
hidden_state = np.zeros((1, 64), dtype=np.float32)

gain_output, hidden_state_out = session.run(
    ["gain_output", "hidden_state_out"],
    {
        "input_frame": input_frame,
        "hidden_state_in": hidden_state,
    }
)

print("\nONNX Runtime inference: PASSED")
print("Gain output shape:", gain_output.shape)
print("Hidden-state output shape:", hidden_state_out.shape)
print("First five gain values:", gain_output[0, :5])
print("Gain range:", float(gain_output.min()), "to", float(gain_output.max()))