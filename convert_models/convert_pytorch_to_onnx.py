import sys
sys.path.append('..')

import os

import torch
import torch.nn as nn
from ultralytics import YOLO


model_dir = "../models/"
torch_model_dir = "pytorch_model"
onnx_model_dir = "onnx_model"
model_name = "yolo11m_ReLU"
input_model_path = os.path.join(model_dir, torch_model_dir, model_name + ".pt")
output_model_path = os.path.join(model_dir, onnx_model_dir, model_name + ".onnx")

model = torch.load(input_model_path)

model.eval()

# モデルの重みをfloat32に変換
model = model.float()

# ダミー入力もfloat32にする
dummy_input = torch.randn(1, 3, 640, 640).float()

# onnx_model_path = os.path.join(model_dir, model_name + ".onnx")

torch.onnx.export(model, dummy_input, output_model_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'], 
                  output_names=['output'])

# 保存
print("Model saved as " + output_model_path)