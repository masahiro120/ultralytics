import os
import onnx
from onnx_tf.backend import prepare

import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

model_dir = "../models/"
# tensorflow_model_dir = "tensorflow_model"
# model_name = "yolo11m_ReLU_cut"
tensorflow_model_dir = "pytorch_model"
model_name = "640_640"
model_path = os.path.join(model_dir, tensorflow_model_dir, model_name)

saved_model_path = os.path.join(model_path, "yolo11m_ReLU.pt.pb")

model = tf.saved_model.load(saved_model_path)
output_model_path = os.path.join(model_path, model_name + ".tflite")



from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# 3. Signature の Concrete Function を取得
concrete_func = list(model.signatures.values())[0]

# 4. Concrete Function から Frozen Graph を作成
frozen_func = convert_variables_to_constants_v2(concrete_func)
frozen_graph = frozen_func.graph

print(f"{len(frozen_graph.as_graph_def().node)} ops in the frozen graph.")  # 作成したFrozen Graphのノード数を表示

# 5. Frozen Graph を .pb ファイルに保存
tf.io.write_graph(
    frozen_graph,
    model_path,  # 保存するディレクトリ
    "frozen_model.pb",  # ファイル名
    as_text=False  # バイナリ形式で保存
)

print("Frozen model saved successfully.")




# # TensorFlowモデルの読み込み
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# # 最適化（オプション）
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # TFLiteモデルに変換して保存
# tflite_model = converter.convert()

# # TFLiteモデルをファイルに保存
# with open(output_model_path, "wb") as f:
#     f.write(tflite_model)