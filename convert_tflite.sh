#!/bin/bash

dataset_path='ultralytics/cfg/datasets/coco8.yaml'

model_dir='ultralytics/cfg/models'

yolo_version_list=(
    'v5'
    'v6'
    'v8'
    'v10'
    '11'
)

model_size_list=(
    'n'
    's'
    'm'
    'l'
    'x'
)

model_list=()
model_name_list=()

for yolo_version in "${yolo_version_list[@]}"; do
    for model_size in "${model_size_list[@]}"; do
        model_name_list+=("yolo$yolo_version$model_size")
        model_list+=("$model_dir/$yolo_version/yolo$yolo_version$model_size.yaml")
    done
done

for i in "${!model_list[@]}"; do
    echo "Model: ${model_list[$i]}"
    echo "Model Name: ${model_name_list[$i]}"
    # python default_convert.py --model_path ${model_list[$i]} --dataset_path $dataset_path
    saved_model_path="${model_name_list[$i]}_saved_model"
    onnx_model_path="${model_name_list[$i]}.onnx"
    echo "Saved Model Path: $saved_model_path"
    mv $saved_model_path tflite_models
    mv $onnx_model_path onnx_models
done