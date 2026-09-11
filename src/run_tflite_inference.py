import numpy as np
import tensorflow as tf


MODEL_PATH = "data/processed/suppression_model_int8.tflite"

N_BANDS = 16
HIDDEN = 64


def find_tensor_by_shape(tensor_details, expected_shape, label):
    """
    Find one TFLite input/output tensor based on its fixed expected shape.
    """
    matches = [
        item for item in tensor_details
        if tuple(item["shape"]) == expected_shape
    ]

    if len(matches) != 1:
        found = [
            (item["name"], tuple(item["shape"]))
            for item in tensor_details
        ]

        raise ValueError(
            f"Could not uniquely identify {label}. "
            f"Expected shape {expected_shape}; found: {found}"
        )

    return matches[0]


def quantize_to_int8(values, tensor_info):
    """
    Convert float32 data into int8 using this tensor's scale and zero point.
    """
    scale, zero_point = tensor_info["quantization"]

    if scale == 0:
        raise ValueError(
            f"Invalid quantization scale for {tensor_info['name']}"
        )

    quantized = np.round(values / scale + zero_point)

    return np.clip(
        quantized,
        -128,
        127
    ).astype(np.int8)


def dequantize_from_int8(values, tensor_info):
    """
    Convert int8 TFLite values back into float32.
    """
    scale, zero_point = tensor_info["quantization"]

    if scale == 0:
        raise ValueError(
            f"Invalid quantization scale for {tensor_info['name']}"
        )

    return (
        values.astype(np.float32) - zero_point
    ) * scale


def main():
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Inputs detected:")
    for item in input_details:
        print(
            f"  {item['name']} | "
            f"shape={item['shape']} | "
            f"type={item['dtype']} | "
            f"quantization={item['quantization']}"
        )

    print("\nOutputs detected:")
    for item in output_details:
        print(
            f"  {item['name']} | "
            f"shape={item['shape']} | "
            f"type={item['dtype']} | "
            f"quantization={item['quantization']}"
        )

    # Identify inputs by their known fixed shapes.
    input_frame_info = find_tensor_by_shape(
        input_details,
        (1, N_BANDS),
        "input frame"
    )

    hidden_in_info = find_tensor_by_shape(
        input_details,
        (1, HIDDEN),
        "hidden-state input"
    )

    # Identify outputs by their known fixed shapes.
    gain_info = find_tensor_by_shape(
        output_details,
        (1, N_BANDS),
        "gain output"
    )

    hidden_out_info = find_tensor_by_shape(
        output_details,
        (1, HIDDEN),
        "hidden-state output"
    )

    # Example streaming inputs:
    # One 16-band log-energy feature frame and an initial zero GRU state.
    input_frame_float = np.ones(
        (1, N_BANDS),
        dtype=np.float32
    )

    hidden_in_float = np.zeros(
        (1, HIDDEN),
        dtype=np.float32
    )

    input_frame_int8 = quantize_to_int8(
        input_frame_float,
        input_frame_info
    )

    hidden_in_int8 = quantize_to_int8(
        hidden_in_float,
        hidden_in_info
    )

    interpreter.set_tensor(
        input_frame_info["index"],
        input_frame_int8
    )

    interpreter.set_tensor(
        hidden_in_info["index"],
        hidden_in_int8
    )

    interpreter.invoke()

    gain_int8 = interpreter.get_tensor(
        gain_info["index"]
    )

    hidden_out_int8 = interpreter.get_tensor(
        hidden_out_info["index"]
    )

    gain_float = dequantize_from_int8(
        gain_int8,
        gain_info
    )

    hidden_out_float = dequantize_from_int8(
        hidden_out_int8,
        hidden_out_info
    )

    print("\nTFLite inference: SUCCESS")
    print("Gain output shape:", gain_float.shape)
    print("Hidden-state output shape:", hidden_out_float.shape)
    print("First five gain values:", gain_float[0, :5])
    print(
        "Gain range:",
        f"{gain_float.min():.4f}",
        "to",
        f"{gain_float.max():.4f}"
    )
    print(
        "Hidden-state range:",
        f"{hidden_out_float.min():.4f}",
        "to",
        f"{hidden_out_float.max():.4f}"
    )


if __name__ == "__main__":
    main()