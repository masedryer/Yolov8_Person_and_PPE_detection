import os
import torch
from ultralytics import YOLO  # Assuming the YOLO model is in PyTorch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# Define paths to save the converted models
yolo_model_path = r"D:\LTA\AI Model\PPE Detection\Yolov8_Person_and_PPE_detection\weights\ppe.pt"
onnx_model_path = os.path.join(os.getcwd(), "yolo_model.onnx")
tf_model_dir = os.path.join(os.getcwd(), "tf_model")
tflite_model_path = os.path.join(os.getcwd(), "yolo_model.tflite")

# Step 1: Load the YOLO model
model = YOLO(yolo_model_path)

# Step 2: Convert the YOLO model to ONNX
dummy_input = torch.randn(1, 3, 640, 640)  # Adjust size according to your model input size
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)

print(f"ONNX model saved at: {onnx_model_path}")

# Step 3: Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Step 4: Convert ONNX model to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_dir)

print(f"TensorFlow model saved at: {tf_model_dir}")

# Step 5: Convert TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
tflite_model = converter.convert()

# Save TFLite model to the current working directory
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_model_path}")
