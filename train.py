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
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--save_dir", type=str, default="", help="Save directory")
# parser.add_argument("--cpu_list", type=int, nargs="+", default=[0], help="List of CPU cores to use")

args = parser.parse_args()

print(args)

# Load the YOLO model
# model = YOLO(args.model_path, task="detect", save_dir=args.save_dir)
save_dir = args.save_dir if args.save_dir != "" else None
print(save_dir)
model = YOLO(args.model_path, task="detect", save_dir=save_dir)
# print(model.model)


# def replace_swish_with_relu(module):
#     for name, child in module.named_children():
#         # SwishをReLUに置き換え
#         if isinstance(child, nn.SiLU):  # PyTorchではSiLUがSwishと同等
#             setattr(module, name, nn.ReLU())
#         else:
#             replace_swish_with_relu(child)  # 再帰的に子モジュールを探索

# モデルのSwishをReLUに置き換え
# replace_swish_with_relu(model.model)

print(model.model)
results = model.train(data=args.dataset_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size, resume=args.resume, save_dir=save_dir)



# results = model.train(data=args.dataset_path, epochs=args.epochs, imgsz=args.imgsz)

# metrics = model.val()

# model.ckpt = results.best if results.best else results.last

# bestまたはlastの重みを取得して設定
# if model.trainer.best.exists():
#     model.ckpt = torch.load(model.trainer.best)
# elif model.trainer.last.exists():
#     model.ckpt = torch.load(model.trainer.last)
# else:
#     print("No valid checkpoint found.")

# print(model.model)  # モデルの内容を確認
# print(model.ckpt)

# model.save("yolov6n.pt")

# resuls = model.benchmark_modify(data=args.dataset_path, imgsz=args.imgsz, half=args.half, int8=args.int8)

# print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))

# model.export(format="onnx", int8=True, simplify=True)
# model.export(format="pb", simplify=True)
# model.export(format="tflite", int8=args.int8, data=args.dataset_path, simplify=True)