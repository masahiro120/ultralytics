import onnx
from onnx import helper, TensorProto
import yaml
import argparse

def load_target_nodes_from_yaml(yaml_file):
    """YAMLファイルから削除するノード名のリストを取得する"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('target_node_names', [])

def remove_nodes_and_dependencies(onnx_file_path, target_node_names, output_path):
    """指定されたノードとその依存ノードを削除する"""
    model = onnx.load(onnx_file_path)
    graph = model.graph

    nodes_to_remove = set()
    outputs_to_check = set()

    for node in graph.node:
        if node.name in target_node_names or any(output in outputs_to_check for output in node.input):
            nodes_to_remove.add(node.name)
            outputs_to_check.update(node.output)

    new_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    onnx.save(model, output_path)
    print(f"指定されたノードが削除されました。新しいモデルは '{output_path}' に保存されました。")

def add_outputs_to_model(model_path, output_node_list, output_name_list, output_path):
    """指定されたノードに新しい出力を追加する"""
    model = onnx.load(model_path)
    graph = model.graph

    for node_name, output_name in zip(output_node_list, output_name_list):
        # ノードを検索
        matching_nodes = [n for n in graph.node if n.name == node_name]
        if not matching_nodes:
            print(f"警告: ノード '{node_name}' が見つかりませんでした。")
            continue
        
        node = matching_nodes[0]

        # ノードの出力テンソルを取得して新しい出力として追加
        for output in node.output:
            output_tensor = helper.ValueInfoProto()
            output_tensor.name = output_name
            output_tensor.type.tensor_type.elem_type = TensorProto.FLOAT
            output_tensor.type.tensor_type.shape.dim.add().dim_param = "?"

            graph.output.append(output_tensor)

    onnx.save(model, output_path)
    print(f"新しい出力が追加されました。モデルは '{output_path}' に保存されました。")

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
    # graph.ClearField("output")
    graph.output.extend([new_output])

    # 新しいモデルを保存
    onnx.save(model, new_model_path)
    print(f"モデルが '{new_model_path}' に保存され、出力が更新されました。")

def main():
    parser = argparse.ArgumentParser(description="ONNXモデルの編集ツール")
    parser.add_argument("--onnx_path", type=str, required=True, help="元のONNXモデルのパス")
    parser.add_argument("--yaml_path", type=str, required=True, help="削除対象ノードが定義されたYAMLファイルのパス")
    parser.add_argument("--output_path", type=str, required=True, help="新しいONNXモデルの保存先パス")

    args = parser.parse_args()

    # YAMLから削除対象ノードのリストを取得
    target_node_names = load_target_nodes_from_yaml(args.yaml_path)

    # ノードを削除
    remove_nodes_and_dependencies(args.onnx_path, target_node_names, args.output_path)

    # 出力を追加するノードと名前のリスト
    output_node_list = ["/model.23/Concat", "/model.23/Concat_1", "/model.23/Concat_2"]
    output_name_list = ["output", "output1", "output2"]

    # 新しい出力を追加
    # add_outputs_to_model(args.output_path, output_node_list, output_name_list, args.output_path)

    for node, name in zip(output_node_list, output_name_list):
        connect_concat_to_output(args.output_path, node, name, args.output_path)

if __name__ == "__main__":
    main()
