from pathlib import Path
import os

import numpy as np
import tensorflow as tf
import torch

from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2
)


N_BANDS = 16
HIDDEN = 64

PYTORCH_MODEL_PATH = Path("data/processed/suppression_model.pt")
FEATURES_PATH = Path("data/processed/X_suppression.npy")

SAVED_MODEL_DIR = Path("data/processed/tf_saved_model")
TFLITE_PATH = Path("data/processed/suppression_model_int8.tflite")


class StreamingSuppressionTF(tf.Module):
    """
    Explicit TensorFlow implementation of the PyTorch GRUCell equations.

    PyTorch gate order:
      reset gate:  r
      update gate: z
      new gate:    n

    Inputs:
      input_frame:      float32, shape (1, 16)
      hidden_state_in:  float32, shape (1, 64)

    Outputs:
      gain_output:      float32, shape (1, 16)
      hidden_state_out: float32, shape (1, 64)
    """

    def __init__(self):
        super().__init__()

        # PyTorch-like GRU weights:
        # W_ih: (3H, input_dim)
        # W_hh: (3H, H)
        # b_ih: (3H,)
        # b_hh: (3H,)
        self.w_ih = tf.Variable(
            tf.zeros((3 * HIDDEN, N_BANDS), dtype=tf.float32),
            trainable=False,
            name="w_ih"
        )

        self.w_hh = tf.Variable(
            tf.zeros((3 * HIDDEN, HIDDEN), dtype=tf.float32),
            trainable=False,
            name="w_hh"
        )

        self.b_ih = tf.Variable(
            tf.zeros((3 * HIDDEN,), dtype=tf.float32),
            trainable=False,
            name="b_ih"
        )

        self.b_hh = tf.Variable(
            tf.zeros((3 * HIDDEN,), dtype=tf.float32),
            trainable=False,
            name="b_hh"
        )

        # Final output layer:
        # PyTorch Linear weights: (16, 64)
        # PyTorch Linear bias: (16,)
        self.out_weight = tf.Variable(
            tf.zeros((N_BANDS, HIDDEN), dtype=tf.float32),
            trainable=False,
            name="out_weight"
        )

        self.out_bias = tf.Variable(
            tf.zeros((N_BANDS,), dtype=tf.float32),
            trainable=False,
            name="out_bias"
        )

        dummy_frame = tf.zeros((1, N_BANDS), dtype=tf.float32)
        dummy_hidden = tf.zeros((1, HIDDEN), dtype=tf.float32)

        self(
            input_frame=dummy_frame,
            hidden_state_in=dummy_hidden
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[1, N_BANDS],
                dtype=tf.float32,
                name="input_frame"
            ),
            tf.TensorSpec(
                shape=[1, HIDDEN],
                dtype=tf.float32,
                name="hidden_state_in"
            )
        ]
    )
    def __call__(self, input_frame, hidden_state_in):
        # PyTorch weight layout: [reset, update, new].
        w_ir, w_iz, w_in = tf.split(self.w_ih, 3, axis=0)
        w_hr, w_hz, w_hn = tf.split(self.w_hh, 3, axis=0)

        b_ir, b_iz, b_in = tf.split(self.b_ih, 3, axis=0)
        b_hr, b_hz, b_hn = tf.split(self.b_hh, 3, axis=0)

        # x @ W.T is required because the stored weight follows
        # the original PyTorch (out_features, in_features) layout.
        r_t = tf.sigmoid(
            tf.matmul(input_frame, w_ir, transpose_b=True)
            + b_ir
            + tf.matmul(hidden_state_in, w_hr, transpose_b=True)
            + b_hr
        )

        z_t = tf.sigmoid(
            tf.matmul(input_frame, w_iz, transpose_b=True)
            + b_iz
            + tf.matmul(hidden_state_in, w_hz, transpose_b=True)
            + b_hz
        )

        n_t = tf.tanh(
            tf.matmul(input_frame, w_in, transpose_b=True)
            + b_in
            + r_t * (
                tf.matmul(hidden_state_in, w_hn, transpose_b=True)
                + b_hn
            )
        )

        hidden_state_out = (
            (1.0 - z_t) * n_t
            + z_t * hidden_state_in
        )

        gain_logits = (
            tf.matmul(
                hidden_state_out,
                self.out_weight,
                transpose_b=True
            )
            + self.out_bias
        )

        gain_output = tf.sigmoid(gain_logits)

        return {
            "gain_output": gain_output,
            "hidden_state_out": hidden_state_out
        }


def load_pytorch_weights_into_tf(tf_model, pytorch_state):
    """
    Copy PyTorch weights directly without reordering or transposing.

    The explicit TensorFlow cell uses the same:
      - gate order: [reset, update, new]
      - matrix layout: (out_features, in_features)
      - bias layout
    as PyTorch.
    """
    tf_model.w_ih.assign(
        pytorch_state["gru.weight_ih_l0"].cpu().numpy().astype(np.float32)
    )

    tf_model.w_hh.assign(
        pytorch_state["gru.weight_hh_l0"].cpu().numpy().astype(np.float32)
    )

    tf_model.b_ih.assign(
        pytorch_state["gru.bias_ih_l0"].cpu().numpy().astype(np.float32)
    )

    tf_model.b_hh.assign(
        pytorch_state["gru.bias_hh_l0"].cpu().numpy().astype(np.float32)
    )

    tf_model.out_weight.assign(
        pytorch_state["out.0.weight"].cpu().numpy().astype(np.float32)
    )

    tf_model.out_bias.assign(
        pytorch_state["out.0.bias"].cpu().numpy().astype(np.float32)
    )


def representative_dataset():
    """
    Provide real input feature frames for TFLite int8 calibration.

    The generator must yield inputs in this order:
      1. input_frame, shape (1, 16)
      2. hidden_state_in, shape (1, 64)
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Calibration features not found: {FEATURES_PATH}\n"
            "Run your dataset-building script first to create "
            "X_suppression.npy."
        )

    features = np.load(FEATURES_PATH).astype(np.float32)

    if len(features) == 0:
        raise ValueError("X_suppression.npy contains no feature frames.")

    rng = np.random.default_rng(42)
    count = min(200, len(features))
    indices = rng.choice(
        len(features),
        size=count,
        replace=False
    )

    for index in indices:
        input_frame = features[index:index + 1]
        hidden_state = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(1, HIDDEN)
        ).astype(np.float32)

        yield [input_frame, hidden_state]


def main():
    if not PYTORCH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained PyTorch model not found: {PYTORCH_MODEL_PATH}"
        )

    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading PyTorch trained weights...")

    pytorch_state = torch.load(
        PYTORCH_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("Building equivalent TensorFlow streaming model...")

    tf_model = StreamingSuppressionTF()

    load_pytorch_weights_into_tf(
        tf_model,
        pytorch_state
    )

    print("Saving TensorFlow SavedModel...")

    tf.saved_model.save(
        tf_model,
        str(SAVED_MODEL_DIR),
        signatures={
            "serving_default": (
                tf_model.__call__.get_concrete_function()
            )
        }
    )

    print(f"Saved TensorFlow model to: {SAVED_MODEL_DIR}")

    print("\nFreezing TensorFlow variables into constants...")

    concrete_function = tf_model.__call__.get_concrete_function()

    frozen_function = convert_variables_to_constants_v2(
        concrete_function
    )

    print("Frozen input tensors:")
    for tensor in frozen_function.inputs:
        print(f"  {tensor.name}: {tensor.shape}, {tensor.dtype}")

    print("Frozen output tensors:")
    for tensor in frozen_function.outputs:
        print(f"  {tensor.name}: {tensor.shape}, {tensor.dtype}")

    print("\nConverting frozen model to full int8 TFLite...")

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [frozen_function],
        tf_model
    )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024

    print(f"\nSaved TFLite model: {TFLITE_PATH}")
    print(f"Model size: {size_kb:.1f} KB")
    print("Quantization: full int8 input, weights, activations, and output.")


if __name__ == "__main__":
    main()