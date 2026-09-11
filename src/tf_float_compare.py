import numpy as np
import tensorflow as tf
import torch

from convert_to_tflite import (
    StreamingSuppressionTF,
    load_pytorch_weights_into_tf,
    PYTORCH_MODEL_PATH
)


N_BANDS = 16
HIDDEN = 64


def main():
    print("Loading PyTorch checkpoint into explicit TensorFlow model...")

    pytorch_state = torch.load(
        PYTORCH_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    tf_model = StreamingSuppressionTF()

    load_pytorch_weights_into_tf(
        tf_model,
        pytorch_state
    )

    inputs = np.load("data/processed/eq_check_inputs.npy")
    pt_gains = np.load("data/processed/eq_check_pt_gains.npy")
    pt_hiddens = np.load(
        "data/processed/eq_check_pt_hiddens_out.npy"
    )

    tf_gains = []
    tf_hiddens = []

    hidden = np.zeros(
        (1, HIDDEN),
        dtype=np.float32
    )

    for frame in inputs:
        frame = frame.reshape(1, N_BANDS).astype(np.float32)

        outputs = tf_model(
            input_frame=tf.constant(frame),
            hidden_state_in=tf.constant(hidden)
        )

        gain = outputs["gain_output"].numpy()
        hidden = outputs["hidden_state_out"].numpy()

        tf_gains.append(gain.flatten())
        tf_hiddens.append(hidden.flatten())

    tf_gains = np.array(tf_gains)
    tf_hiddens = np.array(tf_hiddens)

    gain_diff = np.abs(tf_gains - pt_gains)
    hidden_diff = np.abs(tf_hiddens - pt_hiddens)

    print("\n=== PyTorch vs explicit TensorFlow float32 ===")
    print(
        f"Gain   — max abs diff: {gain_diff.max():.8f} "
        f"| mean abs diff: {gain_diff.mean():.8f}"
    )
    print(
        f"Hidden — max abs diff: {hidden_diff.max():.8f} "
        f"| mean abs diff: {hidden_diff.mean():.8f}"
    )

    print("\nFrame 0 comparison:")
    print("PyTorch gain:", pt_gains[0, :5])
    print("TF gain:     ", tf_gains[0, :5])

    print("\nFinal-frame comparison:")
    print("PyTorch gain:", pt_gains[-1, :5])
    print("TF gain:     ", tf_gains[-1, :5])


if __name__ == "__main__":
    main()