import tensorflow as tf

# 1. Frozen Model (.pb) ファイルのパスを指定
frozen_model_path = "../models/pytorch_model/640_640/frozen_model.pb"
logdir = "../models/pytorch_model/640_640/logdir"  # TensorBoardのログディレクトリ

# 2. Frozen Graphを読み込み
with tf.io.gfile.GFile(frozen_model_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# 3. TensorBoard用にグラフを保存
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    writer = tf.compat.v1.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)
    writer.close()

print(f"Graph saved to TensorBoard log directory: {logdir}")