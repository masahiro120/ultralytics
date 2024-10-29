import sys
sys.path.append('..')

import os

import torch
import torch.nn as nn
from ultralytics import YOLO

def replace_activation(model, old_activation, new_activation):
    """
    モデル内の活性化関数を再帰的に置き換える関数。
    """
    for name, module in model.named_children():
        if isinstance(module, old_activation):
            setattr(model, name, new_activation())
        else:
            replace_activation(module, old_activation, new_activation)

model_dir = "../models/"
torch_model_dir = "pytorch_model"
model_name = "yolo11m"
input_model_path = os.path.join(model_dir, torch_model_dir, model_name + ".pt")

output_model_name = model_name + "_ReLU"
output_model_path = os.path.join(model_dir, torch_model_dir, output_model_name + ".pt")


# model_name = "yolov5s"
# model = YOLO("ultralytics/cfg/models/v5/" + model_name + ".yaml")



model = torch.load(input_model_path)['model']

# print("Model structure:")
# print(model)

# torch.save(model, 'models/yolo11m_model.pt')
# print("Model saved as yolo11m_model.pt")

replace_activation(model, nn.SiLU, nn.ReLU)
print("Model structure after replacing SiLU with ReLU:")
print(model)

torch.save(model, output_model_path)
print("Model saved as yolo11m_model_ReLU.pt")