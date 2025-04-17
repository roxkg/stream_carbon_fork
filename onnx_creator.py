import onnx
from onnx import shape_inference
model = onnx.load("resnet50.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "resnet50.onnx")