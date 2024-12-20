from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("models/yolo11n.pt")

# Export the model to ONNX format
model.export(format="engine")  # creates 'yolo11n.onnx'
