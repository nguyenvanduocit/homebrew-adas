from ultralytics import SAM
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

# Add platform detection
PLATFORM = platform.machine()
IS_ARM = PLATFORM in ['arm64', 'aarch64']

# Optimize for different platforms
if torch.cuda.is_available():
    if IS_ARM:
        # Jetson Nano specific optimizations
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        # Regular CUDA optimizations
        cudnn.benchmark = True
        torch.cuda.empty_cache()
    gc.collect()

# Disable logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

# Device selection
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif IS_ARM:
    device = 'mps'

# Load SAM model
model = SAM("sam2.1_b.pt")
model.to(device)

def get_video_fps(video_path):
    """Calculate the real FPS of a video file"""
    temp_cap = cv2.VideoCapture(video_path)
    if not temp_cap.isOpened():
        return None
    fps = temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()
    return fps

if len(sys.argv) != 2:
    print("Usage: python sam_detect.py <video_path>")
    sys.exit(1)

# Open video
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
real_fps = get_video_fps(video_path)
prev_time = time.time()

# Calculate resize dimensions
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame")

target_width = 640
aspect_ratio = first_frame.shape[1] / first_frame.shape[0]
target_height = int(target_width / aspect_ratio)

# FPS calculation buffer
fps_buffer = deque(maxlen=30)

# Frame skip logic
target_fps = 30  # Lower target FPS for SAM due to higher computational needs
frame_time = 1.0 / target_fps
frame_skip = max(1, int(real_fps / target_fps))

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

# Define points for SAM to track (example: center of frame)
def get_center_point(frame):
    height, width = frame.shape[:2]
    return [[width//2, height//2]], [1]  # Center point with positive label

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Skip frames to maintain target FPS
    if frame_skip > 1:
        for _ in range(frame_skip - 1):
            cap.read()
    
    if time.time() - prev_time < frame_time:
        continue
    
    # Resize frame
    frame = cv2.resize(frame, (target_width, target_height))

    # Clear CUDA cache periodically
    if frame_skip % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Get points for segmentation (you can modify this based on your needs)
    points, labels = get_center_point(frame)
    
    # Perform SAM segmentation
    with torch.no_grad():
        if IS_ARM:
            frame = np.ascontiguousarray(frame)
        results = model(frame, points=points, labels=labels)
    
    # Plot results
    annotated_frame = results[0].plot()
    
    # Memory cleanup
    del results
    if IS_ARM and frame_skip % 30 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    elif frame_skip % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate and display performance metrics
    current_time, avg_fps = calculate_performance_metrics(prev_time, fps_buffer)
    prev_time = current_time
    
    performance_text = create_performance_text(avg_fps)
    
    text_size = cv2.getTextSize(performance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 20
    text_y = annotated_frame.shape[0] - 20
    
    cv2.putText(annotated_frame, performance_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("SAM2 Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()