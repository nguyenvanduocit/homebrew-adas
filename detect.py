from ultralytics import YOLO
import numpy as np
import cv2
import time
import psutil
from collections import deque, defaultdict
import torch
import gc
import torch.backends.cudnn as cudnn
import logging
import sys
import platform
import coremltools

# Add platform detection
PLATFORM = platform.machine()
IS_ARM = PLATFORM in ['arm64', 'aarch64']  # True for both M1 and Jetson
IS_MACOS = platform.system() == 'Darwin' and IS_ARM

# Optimize for different platforms
if torch.cuda.is_available():
    if IS_ARM:
        # Jetson Nano specific optimizations
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False  # Disable for Jetson
        torch.backends.cudnn.deterministic = True
    else:
        # Regular CUDA optimizations
        cudnn.benchmark = True
        torch.cuda.empty_cache()
    gc.collect()

# Before loading the model, add these lines to disable logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

# Update model loading with device selection
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif IS_MACOS:
    device = 'mps'  # Metal Performance Shaders for M1
    # Configure CoreML settings
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.enable_memory_efficient_concat = True

model = YOLO("./models/yolo11n.onnx", task="detect")
if IS_MACOS:
    # Suppress CoreML warnings
    logging.getLogger("coremltools").setLevel(logging.ERROR)
    # Optional: Convert warnings to debug messages
    logging.getLogger("onnxruntime").setLevel(logging.DEBUG)

# Add track history for visualization
track_history = defaultdict(lambda: [])

def get_video_fps(video_path):
    """
    Calculate the real FPS of a video file
    Returns: actual_fps (float)
    """
    temp_cap = cv2.VideoCapture(video_path)
    if not temp_cap.isOpened():
        return None
    
    fps = temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()
    return fps

if len(sys.argv) != 2:
    print("Usage: python detect.py <video_path>")
    sys.exit(1)

# Open webcam
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
real_fps = get_video_fps(video_path)
prev_time = time.time()

# Calculate resize dimensions once
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame")
    
target_width = 640
aspect_ratio = first_frame.shape[1] / first_frame.shape[0]
target_height = int(target_width / aspect_ratio)

# FPS calculation buffer
fps_buffer = deque(maxlen=30)

# Update the frame skip logic:
target_fps = 60  # Set target FPS to 60
frame_time = 1.0 / target_fps  # Time per frame in seconds
frame_skip = max(1, int(real_fps / target_fps))  # Calculate number of frames to skip

print(f"Real FPS: {real_fps}, Frame skip: {frame_skip}")

def calculate_performance_metrics(prev_time, fps_buffer):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    fps_buffer.append(fps)
    avg_fps = sum(fps_buffer) / len(fps_buffer)
    return current_time, avg_fps

def create_performance_text(avg_fps):
    performance_text = f"FPS: {avg_fps:.1f}"
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_reserved() / 1E9
        performance_text += f" | GPU Mem: {gpu_memory:.1f}GB"
    return performance_text

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Skip frames to maintain target FPS
    if frame_skip > 1:
        for _ in range(frame_skip - 1):
            cap.read()  # Skip frames without processing them
    
    # Only process if enough time has passed since last frame
    if time.time() - prev_time < frame_time:
        continue
    
    # Use pre-calculated dimensions
    frame = cv2.resize(frame, (target_width, target_height))

    # Clear CUDA cache periodically (every 100 frames)
    if frame_skip % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Optimize inference based on platform
    with torch.no_grad():
        if IS_ARM:
            # Convert frame to contiguous array for better ARM performance
            frame = np.ascontiguousarray(frame)
        # Add persist=True to maintain tracks between frames
        results = model.track(frame, conf=0.25, iou=0.45, classes=[2, 9], persist=True, max_det=5, vid_stride=1)
    
    annotated_frame = results[0].plot(line_width=1, font_size=8)
    
    # Add track visualization (optional)
    boxes = results[0].boxes.xywh.cpu()
    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:  # Check if id exists and is not None
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:  # Retain 30 frames of history
                track.pop(0)
                
            # Draw tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
    
    # More aggressive memory cleanup for Jetson
    del results
    if IS_ARM and frame_skip % 30 == 0:  # More frequent cleanup on ARM
        torch.cuda.empty_cache()
        gc.collect()
    elif frame_skip % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate performance metrics
    current_time, avg_fps = calculate_performance_metrics(prev_time, fps_buffer)
    prev_time = current_time
    
    # Create performance text
    performance_text = create_performance_text(avg_fps)
    
    text_size = cv2.getTextSize(performance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 20
    text_y = annotated_frame.shape[0] - 20
    
    cv2.putText(annotated_frame, performance_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close windows
cap.release()
cv2.destroyAllWindows()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()