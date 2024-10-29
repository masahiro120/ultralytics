import tensorflow as tf

# TFLiteモデルの読み込み
interpreter = tf.lite.Interpreter(model_path="../models/tensorflow_model/yolo11m_ReLU_cut/yolo11m_ReLU_cut.tflite")
interpreter.allocate_tensors()

# 入力テンソルの情報を取得
input_details = interpreter.get_input_details()
print("Input Shape:", input_details[0]['shape'])

# 出力テンソルの情報を取得
output_details = interpreter.get_output_details()
for i, output_detail in enumerate(output_details):
    print(f"Output Shape {i}:", output_detail['shape'])


# 入力テンソルの形状に従ったデモ入力の生成
demo_input = tf.random.normal(input_details[0]['shape'])

# 入力データをモデルにセット
interpreter.set_tensor(input_details[0]['index'], demo_input)

# 推論の実行
interpreter.invoke()

# 出力を取得して表示
outputs = []
for output_detail in output_details:
    output_data = interpreter.get_tensor(output_detail['index'])
    outputs.append(output_data)
    print(f"Output Data (Shape {output_detail['shape']}):", output_data.shape)