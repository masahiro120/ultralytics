#!/bin/bash

dataset_path='ultralytics/cfg/datasets/coco.yaml'

model_dir='ultralytics/cfg/models'

yolo_version_list=(
    # 'v3'
    # 'v5'
    'v5_6'
    # 'v8'
    # 'v9'
    # 'v10'
    # '11'
)

default_model_size_list=(
    # 'n'
    # 's'
    'm'
    'l'
    'x'
)

v3_model_size_list=(
    'tiny'
    'normal'
    'spp'
)

v9_model_size_list=(
    't'
    's'
    'm'
    'c'
    'e'
)

v10_model_size_list=(
    'n'
    's'
    'm'
    'b'
    'l'
    'x'
)

imgsz=640

output_dir="accuracy_output"
for yolo_version in "${yolo_version_list[@]}"; do
    if [ "$yolo_version" == "v3" ]; then
        model_size_list=("${v3_model_size_list[@]}")
    elif [ "$yolo_version" == "v9" ]; then
        model_size_list=("${v9_model_size_list[@]}")
    elif [ "$yolo_version" == "v10" ]; then
        model_size_list=("${v10_model_size_list[@]}")
    else
        model_size_list=("${default_model_size_list[@]}")
    fi
    output_version_dir="${output_dir}/${yolo_version}"
    mkdir -p $output_version_dir
    for model_size in "${model_size_list[@]}"; do
        output_size_dir="${output_version_dir}/${model_size}"

        if [ "$model_size" == "tiny" ]; then
            model_size="-tiny"
        elif [ "$model_size" == "normal" ]; then
            model_size=""
        elif [ "$model_size" == "spp" ]; then
            model_size="-spp"
        fi

        mkdir -p $output_size_dir
        if [ "$yolo_version" == "v5_6" ]; then
            model_name="yolov5${model_size}6"
            imgsz=1280
        else
            model_name="yolo$yolo_version$model_size"
        fi

        echo "Model Name: $model_name"
        # if [ "$yolo_version" == "v5_6" ]; then
        #     if [ "$model_size" == "m" ] || [ "$model_size" == "l" ] || [ "$model_size" == "x" ]; then
                echo "Float 32"
                output_file="${output_size_dir}/${model_name}_float32.txt"
                python default_convert.py --model_path "$model_name.pt" --dataset_path $dataset_path --imgsz $imgsz > $output_file

                echo "Float 16"
                output_file="${output_size_dir}/${model_name}_float16.txt"
                python default_convert.py --model_path "$model_name.pt" --dataset_path $dataset_path --imgsz $imgsz --half > $output_file
        #     fi
        # fi

        # if [ "$yolo_version" == "v3" ]; then
        #     if [ "$model_size" == "-tiny" ]; then
        #         continue
        #     fi
        # fi

        # if [ "$yolo_version" == "v5" ]; then
        #     if [ "$model_size" == "n" ] || [ "$model_size" == "s" ] || [ "$model_size" == "m" ]; then
        #         continue
        #     fi
        # fi

        # if [ "$yolo_version" == "v5_6" ]; then
        #     if [ "$model_size" == "n" ]; then
        #         continue
        #     fi
        # fi

        # if [ "$yolo_version" == "v8" ]; then
        #     if [ "$model_size" == "n" ] || [ "$model_size" == "s" ] || [ "$model_size" == "m" ]; then
        #         continue
        #     fi
        # fi

        # if [ "$yolo_version" == "v9" ]; then
        #     if [ "$model_size" == "t" ] || [ "$model_size" == "s" ] || [ "$model_size" == "m" ] || [ "$model_size" == "c" ]; then
        #         continue
        #     fi
        # fi

        # if [ "$model_size" != "l" ] && [ "$model_size" != "x" ]; then
        #     echo "Int 8"
        #     output_file="${output_size_dir}/${model_name}_int8.txt"
        #     python default_convert.py --model_path "$model_name.pt" --dataset_path $dataset_path --imgsz $imgsz --int8 > $output_file
        # fi
    done
done
