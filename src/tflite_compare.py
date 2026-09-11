import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="data/processed/suppression_model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Identify tensors by shape, as your handover doc correctly notes is necessary
feat_in = next(d for d in input_details if d['shape'][-1] == 16)
hid_in  = next(d for d in input_details if d['shape'][-1] == 64)
gain_out = next(d for d in output_details if d['shape'][-1] == 16)
hid_out  = next(d for d in output_details if d['shape'][-1] == 64)

inputs = np.load("data/processed/eq_check_inputs.npy")
hiddens_pt_in = np.load("data/processed/eq_check_hiddens_in.npy")
pt_gains = np.load("data/processed/eq_check_pt_gains.npy")
pt_hiddens = np.load("data/processed/eq_check_pt_hiddens_out.npy")

tflite_gains, tflite_hiddens = [], []
hidden = np.zeros((1, 64), dtype=np.float32)  # TFLite runs its OWN streaming state

for i in range(len(inputs)):
    frame = inputs[i].reshape(1, 16).astype(np.float32)

    # Quantize using the model's OWN scale/zero-point (read from the file itself,
    # not hardcoded — robust even if you re-export/re-quantize later)
    f_scale, f_zp = feat_in['quantization']
    h_scale, h_zp = hid_in['quantization']
    frame_q = np.clip(
    np.round(frame / f_scale + f_zp),
    -128,
    127
    ).astype(np.int8)

    hidden_q = np.clip(
    np.round(hidden / h_scale + h_zp),
    -128,
    127
    ).astype(np.int8)

    interpreter.set_tensor(feat_in['index'], frame_q)
    interpreter.set_tensor(hid_in['index'], hidden_q)
    interpreter.invoke()

    gain_q = interpreter.get_tensor(gain_out['index'])
    hnew_q = interpreter.get_tensor(hid_out['index'])

    g_scale, g_zp = gain_out['quantization']
    ho_scale, ho_zp = hid_out['quantization']
    gain_deq = (gain_q.astype(np.float32) - g_zp) * g_scale
    hnew_deq = (hnew_q.astype(np.float32) - ho_zp) * ho_scale

    tflite_gains.append(gain_deq.flatten())
    tflite_hiddens.append(hnew_deq.flatten())
    hidden = hnew_deq.reshape(1, 64)  # carry TFLite's own state forward

tflite_gains = np.array(tflite_gains)
tflite_hiddens = np.array(tflite_hiddens)

gain_diff = np.abs(tflite_gains - pt_gains)
hidden_diff = np.abs(tflite_hiddens - pt_hiddens)
print("\nPer-frame error summary:")
for i in range(len(inputs)):
    print(
        f"Frame {i}: "
        f"gain max={gain_diff[i].max():.4f}, "
        f"gain mean={gain_diff[i].mean():.4f}, "
        f"hidden max={hidden_diff[i].max():.4f}, "
        f"hidden mean={hidden_diff[i].mean():.4f}"
    )
print("=== PyTorch vs TFLite int8 — Numerical Equivalence Check ===")
print(f"Gain   — max abs diff: {gain_diff.max():.4f} | mean abs diff: {gain_diff.mean():.4f}")
print(f"Hidden — max abs diff: {hidden_diff.max():.4f} | mean abs diff: {hidden_diff.mean():.4f}")
print(f"\n(Gains are 0-1 range; diffs under ~0.05 are generally acceptable for int8 quantization.")
print(f" Hidden state diff matters less directly, but large drift compounds over long sequences.)")