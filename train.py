from ultralytics.models.yolo.detect import DetectionTrainer

args = dict(model="yolo11s.pt", data="coco8.yaml", epochs=3)
trainer = DetectionTrainer(overrides=args)
trainer.train()

trainer.export(format="onnx", simplify=True)
