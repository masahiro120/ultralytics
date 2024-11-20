# models=(
#     'yolo11n'
#     'yolo11s'
#     'yolo11m'
#     'yolo11l'
#     'yolo11x'
#     'yolov10n'
#     'yolov10s'
#     'yolov10m'
#     'yolov10b'
#     'yolov10l'
#     'yolov10x'
#     'yolov9t'
#     'yolov9s'
#     'yolov9m'
#     'yolov9c'
#     'yolov9e'
#     'yolov8n'
#     'yolov8s'
#     'yolov8m'
#     'yolov8l'
#     'yolov8x'
#     'yolov5n'
#     'yolov5s'
#     'yolov5m'
#     'yolov5l'
#     'yolov5x'
# )

# Sorted by model size
models=(
    # 'yolo11n'
    # 'yolov10n'
    # 'yolov8n'
    # 'yolov5n'

    # 'yolo11s'
    # 'yolov10s'
    # 'yolov8s'
    # 'yolov5s'
    
    # 'yolo11m'
    # 'yolov10m'
    # 'yolov8m'
    # 'yolov5m'
    
    # 'yolov10b'
    
    # 'yolo11l'
    # 'yolov10l'
    # 'yolov8l'
    # 'yolov5l'
    
    # 'yolo11x'
    # 'yolov10x'
    # 'yolov8x'
    # 'yolov5x'

    'yolov9t'
    'yolov9s'
    # 'yolov9m'
    # 'yolov9c'
    # 'yolov9e'
)

imgsz=640
dataset_path='coco.yaml'
epochs=20
batch_size=16

mkdir -p "train_log"

for model in "${models[@]}"; do
    mkdir -p "train_log/$model"
    echo "Training $model"
    python train.py --model_path "$model.pt" --imgsz $imgsz --epochs $epochs --batch_size $batch_size --dataset_path $dataset_path --save_dir "train_log/$model" > "train_log/$model/$model.log"
    # python train.py --model_path "$model.pt" --imgsz $imgsz --epochs $epochs --batch_size $batch_size --dataset_path $dataset_path --save_dir "train_log/$model"
    cp train_log/$model/$model.log $HOME/OneDrive/ドキュメント/train_log
done

# shutdown -h now