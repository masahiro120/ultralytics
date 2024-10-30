import os
import onnx
from onnx_tf.backend import prepare

model_dir = "../models/"
onnx_model_dir = "onnx_model"
tensorflow_model_dir = "tensorflow_model"
model_name = "yolo11m_ReLU_cut"
input_model_path = os.path.join(model_dir, onnx_model_dir, model_name + ".onnx")
output_model_path = os.path.join(model_dir, tensorflow_model_dir, model_name)

onnx_model = onnx.load(input_model_path)

try:
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    for input in onnx_model.graph.input:
        print(f"Input: {input.name}")

    for output in onnx_model.graph.output:
        print(f"Output: {output.name}")
except onnx.checker.ValidationError as e:
    print(f"ONNX model is invalid: {e}")

tf_rep = prepare(onnx_model)

saved_model_path = os.path.join(output_model_path, "saved_model")

tf_rep.export_graph(saved_model_path)
print(f"TensorFlow model saved at {saved_model_path}")