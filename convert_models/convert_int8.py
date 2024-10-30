from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    model.export(format="tflite", int8=True, data='../ultralytics/cfg/datasets/coco8.yaml')