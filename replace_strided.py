import tensorflow as tf
import numpy as np

# 1. TFLiteモデルをTensorFlow形式に変換する関数
def convert_tflite_to_tf(tflite_model_path, saved_model_dir):
    # TFLiteモデルをロード
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # 入力と出力の詳細を取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # モデルの期待する入力形状を取得
    input_shape = input_details[0]['shape']

    @tf.function
    def model_func(inputs: tf.Tensor):
        # TFLiteモデルにNumPy配列として入力をセット
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()

        # 出力を取得して返す
        output = interpreter.get_tensor(output_details[0]['index'])
        return output

    # Concrete Functionを生成
    concrete_func = model_func.get_concrete_function(
        tf.TensorSpec(input_shape, dtype=tf.float32))

    # TensorFlowのSavedModel形式で保存
    tf.saved_model.save(model_func, saved_model_dir, signatures=concrete_func)
    print(f"Model saved to {saved_model_dir}.")

# 2. TensorFlowモデルで`STRIDED_SLICE`を置き換える
def replace_strided_slice_in_model(saved_model_dir, modified_model_dir):
    # SavedModel形式のモデルをロード
    model = tf.saved_model.load(saved_model_dir)

    @tf.function
    def modified_model(input_tensor: tf.Tensor):
        # `STRIDED_SLICE`の代替スライス操作
        begin = [0, 0, 0, 0]
        size = [-1, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]]
        x = tf.slice(input_tensor, begin, size)

        # 元のモデルで処理
        return model(x)

    # Concrete Functionを再生成
    concrete_func = modified_model.get_concrete_function(
        tf.TensorSpec([1, 640, 640, 3], dtype=tf.float32))

    # 置き換えたモデルを保存
    tf.saved_model.save(modified_model, modified_model_dir, signatures=concrete_func)
    print(f"Modified model saved to {modified_model_dir}.")

# 3. TensorFlowモデルをTFLite形式に再変換する
def convert_tf_to_tflite(saved_model_dir, output_path):
    # SavedModel形式からTFLiteモデルに変換
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # TFLiteモデルをファイルに保存
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_path}")

# 使用例
tflite_model_path = "yolo11s_saved_model/yolo11s_float32.tflite"
saved_model_dir = "saved_model_tf"
modified_model_dir = "saved_model_modified"
output_model_path = "yolo11s_float32_replaced.tflite"

# 1. TFLiteモデルをTensorFlow形式に変換
convert_tflite_to_tf(tflite_model_path, saved_model_dir)

# 2. `STRIDED_SLICE`を置き換えてTensorFlowモデルを再構築
replace_strided_slice_in_model(saved_model_dir, modified_model_dir)

# 3. TensorFlowモデルをTFLite形式に再変換
convert_tf_to_tflite(modified_model_dir, output_model_path)
