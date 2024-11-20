from ultralytics import YOLO
import torch.nn as nn
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="ultralytics/cfg/models/11/yolo11s.yaml", help="YAML file path")
parser.add_argument("--dataset_path", type=str, default="ultralytics/cfg/datasets/coco8.yaml", help="Dataset YAML file path")
parser.add_argument("--half", action="store_true", help="Use half precision")
parser.add_argument("--int8", action="store_true", help="Use int8 precision")
parser.add_argument("--imgsz", type=int, default=640, help="Image size")

args = parser.parse_args()

print(args)

# Load the YOLO model
model = YOLO(args.model_path)

# results = model.train(data=args.dataset_path, epochs=1, imgsz=args.imgsz)

# # model.ckpt = results.best if results.best else results.last

# # bestまたはlastの重みを取得して設定
# if model.trainer.best.exists():
#     model.ckpt = torch.load(model.trainer.best)
# elif model.trainer.last.exists():
#     model.ckpt = torch.load(model.trainer.last)
# else:
#     print("No valid checkpoint found.")

# # print(model.model)  # モデルの内容を確認
# # print(model.ckpt)

# model.save("yolov6n.pt")

resuls = model.benchmark_modify(data=args.dataset_path, imgsz=args.imgsz, half=args.half, int8=args.int8)

# print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))

# model.export(format="onnx", simplify=True)
# model.export(format="pb", simplify=True)
# model.export(format="tflite", int8=args.int8, data=args.dataset_path, simplify=True)