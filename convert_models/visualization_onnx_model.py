import onnx

def print_onnx_model_structure(onnx_file_path):
    # ONNXモデルを読み込む
    model = onnx.load(onnx_file_path)

    # モデルのグラフを取得
    graph = model.graph

    print("== ONNX Model Structure ==")
    print(f"Model Name: {graph.name}")
    print(f"Number of Inputs: {len(graph.input)}")
    print(f"Number of Outputs: {len(graph.output)}")
    print(f"Number of Nodes: {len(graph.node)}\n")

    # 各入力ノードの情報を表示
    print("== Inputs ==")
    for input in graph.input:
        print(f"Name: {input.name}, Type: {input.type.tensor_type.elem_type}, Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
    print()

    # 各出力ノードの情報を表示
    print("== Outputs ==")
    for output in graph.output:
        print(f"Name: {output.name}, Type: {output.type.tensor_type.elem_type}, Shape: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
    print()

    # 各ノード（レイヤー）の情報を表示
    print("== Layers ==")
    for node in graph.node:
        print(f"Name: {node.name}, OpType: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")
    print()

# 使用例
onnx_file_path = "../models/onnx_model/yolo11m_ReLU_cut.onnx"  # ONNXモデルファイルのパスを指定
print_onnx_model_structure(onnx_file_path)
