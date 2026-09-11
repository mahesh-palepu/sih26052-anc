import numpy as np
import tensorflow as tf

MODEL_PATH = "data/processed/suppression_model_int8.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite inputs:")
for item in input_details:
    print(f"  Name: {item['name']}")
    print(f"  Shape: {item['shape']}")
    print(f"  Type: {item['dtype']}")
    print(f"  Quantization: {item['quantization']}")
    print()

print("TFLite outputs:")
for item in output_details:
    print(f"  Name: {item['name']}")
    print(f"  Shape: {item['shape']}")
    print(f"  Type: {item['dtype']}")
    print(f"  Quantization: {item['quantization']}")
    print()

assert len(input_details) == 2, "Expected two inputs."
assert len(output_details) == 2, "Expected two outputs."

assert all(
    item["dtype"] == np.int8
    for item in input_details
), "Expected int8 TFLite inputs."

assert all(
    item["dtype"] == np.int8
    for item in output_details
), "Expected int8 TFLite outputs."

print("SUCCESS: Model has 2 int8 inputs and 2 int8 outputs.")