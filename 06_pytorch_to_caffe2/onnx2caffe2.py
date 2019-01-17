import onnx
from caffe2.python.onnx.backend import Caffe2Backend
import argparse

parser = argparse.ArgumentParser(description="Convert ONNX models "
    "to Caffe2 models")

parser.add_argument("--onnx-model", required=True,
    help="The ONNX model")
parser.add_argument("--caffe2-init", required=True,
    help="The output file for the caffe2 model init file. ")
parser.add_argument("--caffe2-predict", required=True,
    help="The output file for the caffe2 model predict file. ")


if __name__ == "__main__":
    args = parser.parse_args()
    onnx = onnx.load(args.onnx_model)
    caffe2_init, caffe2_predict = \
        Caffe2Backend.onnx_graph_to_caffe2_net(onnx, device="CUDA")
    caffe2_init_str = caffe2_init.SerializeToString()
    with open(args.caffe2_init, "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open(args.caffe2_predict, "wb") as f:
        f.write(caffe2_predict_str)
    with open("onnx-predict.pbtxt", "w") as f:
        f.write(str(caffe2_predict))
