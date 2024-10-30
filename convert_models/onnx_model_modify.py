import onnx
from onnx import helper, TensorProto

def remove_nodes_and_dependencies(onnx_file_path, target_node_names, output_path):
    # ONNXモデルを読み込む
    model = onnx.load(onnx_file_path)
    graph = model.graph

    # 削除対象ノードとその依存ノードのセット
    nodes_to_remove = set()
    outputs_to_check = set()

    # ターゲットノードの検出と依存ノードの収集
    for node in graph.node:
        # ターゲットノード名に含まれるか、依存するノードかをチェック
        if node.name in target_node_names or any(output in outputs_to_check for output in node.input):
            nodes_to_remove.add(node.name)
            outputs_to_check.update(node.output)

    # ノードをフィルタリングして残すノードだけ保持
    new_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # 新しいONNXモデルを保存
    onnx.save(model, output_path)
    print(f"指定されたノードおよび依存ノードが削除されました。新しいモデルは '{output_path}' に保存されました。")

def connect_concat_to_output(onnx_file_path, concat_node_name, new_output_name, new_model_path):
    # モデルをロード
    model = onnx.load(onnx_file_path)
    graph = model.graph

    # Concatノードの出力を取得
    concat_node = None
    for node in graph.node:
        if node.name == concat_node_name:
            concat_node = node
            break

    if concat_node is None:
        raise ValueError(f"指定されたConcatノード '{concat_node_name}' が見つかりませんでした。")

    # Concatの出力名を取得
    concat_output_name = concat_node.output[0]

    # 新しい出力を作成
    new_output = helper.make_tensor_value_info(
        new_output_name,  # 出力名
        TensorProto.FLOAT,  # データ型
        None  # 必要に応じて具体的な形状を指定
    )

    # 出力ノードの接続を更新
    new_output.name = concat_output_name  # Concatノードの出力名と一致させる

    # 既存の出力をクリアし、新しい出力を設定
    graph.ClearField("output")
    graph.output.extend([new_output])

    # 新しいモデルを保存
    onnx.save(model, new_model_path)
    print(f"モデルが '{new_model_path}' に保存され、出力が更新されました。")

def remove_specific_outputs(onnx_file_path, outputs_to_remove, output_path):
    # ONNXモデルを読み込む
    model = onnx.load(onnx_file_path)
    graph = model.graph

    # 指定された出力名を削除
    new_outputs = [output for output in graph.output if output.name not in outputs_to_remove]

    # 出力を更新
    graph.ClearField("output")
    graph.output.extend(new_outputs)

    # 新しいONNXモデルを保存
    onnx.save(model, output_path)
    print(f"指定された出力が削除されました。新しいモデルは '{output_path}' に保存されました。")

# 使用例
onnx_file_path = "../models/onnx_model/yolo11m_ReLU.onnx"  # 元のモデルのパス
target_node_names = [
    "/model.23/Split",
    "/model.23/Expand",
    "/model.23/Expand_1",
    "/model.23/Expand_2",
    "/model.23/Expand_3",
    "/model.23/Expand_4",
    "/model.23/Expand_5",
    "/model.23/Constant_3",
    "/model.23/Constant_4",
    "/model.23/Constant_5",
    "/model.23/Constant_6",
    "/model.23/Constant_7",
    "/model.23/Constant_8",
    "/model.23/Constant_9",
    "/model.23/Constant_10",
    "/model.23/Constant_11",
    "/model.23/Constant_12",
    "/model.23/Constant_13",
    "/model.23/Constant_14",
    "/model.23/Constant_15",
    "/model.23/Constant_16",
    "/model.23/Constant_17",
    "/model.23/Constant_18",
    "/model.23/Constant_19",
    "/model.23/Constant_20",
    "/model.23/Constant_21",
    "/model.23/Constant_22",
    "/model.23/dfl/Constant",
    "/model.23/dfl/Constant_1",

    "/model.23/Concat_3",

    "/model.23/Reshape",
    "/model.23/Reshape_1",
    "/model.23/Reshape_2",

    "/model.23/Constant",
    "/model.23/Constant_1",
    "/model.23/Constant_2",
]  # 削除したいノード名のリスト

output_path = "../models/onnx_model/yolo11m_ReLU_cut.onnx"  # 保存する新しいモデルのパス

concat_node_name = "/model.23/Concat_3"  # Concatノードの名前（画像の例で適切に指定）
output_name = "final_output"  # 新しい出力名

remove_nodes_and_dependencies(onnx_file_path, target_node_names, output_path)

remove_specific_outputs(output_path, ["output"], output_path)

# new_model_path = "../models/onnx_model/yolo11m_ReLU_with_output.onnx"  # 新しいモデルの保存先

# connect_concat_to_output(output_path, concat_node_name, output_name, new_model_path)


