import sys
sys.path.append('..')
sys.path.append('../..')

import os
import cv2
import tensorflow as tf
import torch

from convert import model_convert

from ultralytics import YOLO


CHECKPOINT_PATH = "../../models/pytorch_model/yolo11m_ReLU.pt"
DO_OPTIMIZE = True
OPTIMIZATIONS = [tf.lite.Optimize.DEFAULT]

print("CHECKPOINT_PATH: ", CHECKPOINT_PATH)
print("DO_OPTIMIZE: ", DO_OPTIMIZE)
print("OPTIMIZATIONS: ", OPTIMIZATIONS)

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT =  640
INPUT_SHAPE = (1, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)

IMG_PATH = "../sample/imgs/0020.jpg"

def infer(tflite_filepath, img_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_filepath)
    interpreter.allocate_tensors()  # allocate memory

    # 入力層の構成情報を取得する
    input_details = interpreter.get_input_details()

    # 入力層に合わせて、画像を変換する
    img = cv2.imread(img_path)
    img = (img - 128) / 256  # 明度を正規化（モデル学習時のデータ前処理に合わせる）
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_data = cv2.resize(img, (input_shape[2], input_shape[3])).transpose(
        (2, 0, 1)).reshape(input_shape).astype(input_dtype)
    # indexにテンソルデータのポインタをセット
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 推論実行
    interpreter.invoke()

    # 出力層から結果を取り出す
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if __name__ == "__main__":
    CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
    CHECKPOINT_BASE = os.path.basename(CHECKPOINT_PATH)
    OUT_DIR = os.path.join(CHECKPOINT_DIR, f"{MODEL_INPUT_WIDTH}_{MODEL_INPUT_HEIGHT}")
    os.makedirs(OUT_DIR, exist_ok=True)

    onnx_filepath = os.path.join(OUT_DIR, f"{CHECKPOINT_BASE}.onnx")
    pb_filepath = os.path.join(OUT_DIR, f"{CHECKPOINT_BASE}.pb")
    tflite_filepath = os.path.join(OUT_DIR, f"{CHECKPOINT_BASE}.tflite")

    # convert model
    # model = load_model.load_model_checkpoint(CHECKPOINT_PATH, use_cuda=False) # 変換したいモデルに合わせる

    model = torch.load(CHECKPOINT_PATH)

    model.eval()  # 推論モードに切り替え

    # モデルの重みが HalfTensor の場合、入力も half() に変換
    if next(model.parameters()).dtype == torch.float16:
        dummy_input = torch.randn(INPUT_SHAPE).half()
        model = model.half()
    else:
        dummy_input = torch.randn(INPUT_SHAPE)


    # 入力の型を確認
    print("input dtype: ", dummy_input.dtype)

    # モデルの型を確認
    print("model dtype: ", next(model.parameters()).dtype)

    dummy_input = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    model = model.to(dtype=torch.float32)

    print("Updated input dtype: ", dummy_input.dtype)
    print("Updated model dtype: ", next(model.parameters()).dtype)

    tflite_filepath = model_convert.convert(
        model, onnx_filepath, pb_filepath, tflite_filepath,
        INPUT_SHAPE, DO_OPTIMIZE, OPTIMIZATIONS,
    )

    # test
    output_data = infer(tflite_filepath, IMG_PATH)
    print(output_data)