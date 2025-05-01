
#########################################
#   INSTALL AND SET DEPENDENCIES        #
#########################################

import os
# os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "debug" 
import cv2
import time
import torch
import torchreid
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, simpledialog
import torch.multiprocessing as mp
import logging
import json
import requests
import shutil
import re
from io import BytesIO
import threading
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Empty  # Explicitly import the correct Empty exception
from torch.amp import autocast
import boto3
from botocore.exceptions import ClientError
import zipfile
import tempfile

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
logger = logging.getLogger(__name__)

# Configure S3 upload logging
s3_logger = logging.getLogger('s3_upload')
s3_logger.setLevel(logging.INFO)
s3_handler = logging.FileHandler('s3_upload.log')
s3_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
s3_logger.addHandler(s3_handler)
s3_logger.addHandler(logging.StreamHandler())

# Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Simulated user database (replace with a real database in production)
USERS = {"admin": "admin"}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# Initialize YOLO model
try:
    model = YOLO("yolo11n_12-2.engine", task='detect')
    # logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

telegram_pool = ThreadPoolExecutor(max_workers=5)

# Initialize ReID model
reid_model = torchreid.models.build_model(
    name='osnet_ibn_x1_0',
    num_classes=1000,
    pretrained=True
).to(device)
reid_model.eval()

# Preprocessing function for crops
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]).cuda()
])

# Clear GPU memory
torch.cuda.empty_cache()

# AWS S3 configuration
BUCKET_NAME = 'ip-cameras-recordings'  # Replace with your S3 bucket name
RECORDINGS_DIR = './recordings'  # Local directory with recordings
# UPLOAD_INTERVAL = 12 * 3600  # 12 hours in seconds
# RETENTION_PERIOD = 48 * 3600  # 48 hours in seconds

UPLOAD_INTERVAL = 1 * 3600  # 1 hours in seconds
RETENTION_PERIOD = 4 * 3600  # 4 hours in seconds

#########################################
#   GLOBAL VARIABLES                    #
#########################################

batch_duration = 10 * 60  # 30 minutes in seconds

database_dir = './body_database'
recordings_dir = './recordings'  # New directory for video recordings

# Global dictionary to store latest frames for each camera
latest_frames = {}

# Global dictionary to store latest raw frames and detections
latest_raw_frames = {}  # Added for raw frames
latest_detections = {}  # Added for detection results

# Global variable to store frames for queue
frame_queues = []

# URLs of all IP cameras
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.122:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.138:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.118:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.116:554/Streaming/Channels/101?transport=tcp",
    # "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp",
]

# Credentials for Telegram Channel/Group
bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
chat_id = '-1002599945824'

# Global variable to cache active cameras
active_cameras_cache = list(range(len(camera_urls)))  # Default to all cameras initially
cache_lock = threading.Lock()

# Track last alert time per track_id
last_alert_time = {}  

# Global variables
database_embeddings = None  # Tensor of all embeddings
database_names = []  # Corresponding names

# Global variable to store processes (moved from main block to global scope)
processes = []

##############################
#   HELPER FUNCTIONS         #
##############################

def add_person_to_database(name, images):
    person_dir = os.path.join(database_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    embeddings = []
    for idx, image in enumerate(images):
        image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
        cv2.imwrite(image_filename, image)
        embeddings.append(extract_embedding(image))
    return embeddings

def extract_embedding(image):
    try:
        if image.size == 0:
            logger.error("Empty image passed to extract_embedding")
            return np.zeros(1000)  # Return dummy embedding if image is invalid
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast(device_type='cuda'):
                embedding = reid_model(image_tensor)
        embedding = embedding.squeeze(0).cpu().numpy()
        # logger.debug(f"Embedding generated: shape={embedding.shape}, norm={np.linalg.norm(embedding):.2f}")
        return embedding
    except Exception as e:
        logger.error(f"Error in extract_embedding: {str(e)}")
        return np.zeros(1000)  # Fallback

def match_embedding(query_embedding, database, shared_namespace, threshold=0.7):
    if shared_namespace.database_embeddings is None or len(shared_namespace.database_embeddings) == 0:
        logger.debug("Database embeddings empty or not initialized")
        return None, 0
    try:
        # Move database_embeddings to GPU only when needed
        db_embeddings = shared_namespace.database_embeddings.to(device)
        query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0).to(device)
        with torch.no_grad():
            similarities = torch.cosine_similarity(query_tensor, db_embeddings)
            best_similarity, best_idx = torch.max(similarities, dim=0)
            best_similarity = best_similarity.item()
            top_similarities = torch.topk(similarities, min(3, len(similarities))).values.tolist()
            # logger.debug(f"Top similarities: {top_similarities}, Best: {best_similarity:.2f}, Threshold: {threshold}")
            if best_similarity >= threshold:
                best_match = shared_namespace.database_names[best_idx]
                # logger.info(f"Match found: {best_match} with similarity {best_similarity:.2f}")
                return best_match, best_similarity
        return None, 0
    except Exception as e:
        logger.error(f"Error in match_embedding: {str(e)}")
        return None, 0

def load_database(shared_namespace):
    database = {}
    embeddings_list = []
    names_list = []
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                image_count = 0
                for image_filename in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        embedding = extract_embedding(image)
                        embeddings.append(embedding)
                        embeddings_list.append(embedding)
                        names_list.append(person_name)
                        image_count += 1
                    else:
                        logger.warning(f"Failed to load image: {image_path}")
                database[person_name] = embeddings
                # logger.info(f"Loaded {image_count} images for {person_name}")
        if embeddings_list:
            # Keep tensor on CPU to avoid CUDA serialization issues
            shared_namespace.database_embeddings = torch.tensor(np.stack(embeddings_list))  # No .to(device) here
            shared_namespace.database_names = names_list
            # logger.info(f"Database loaded: {len(shared_namespace.database_names)} embeddings, {len(database)} unique persons")
        else:
            logger.warning("No embeddings loaded into database_embeddings")
    else:
        logger.error(f"Database directory {database_dir} does not exist")
    return database, embeddings_list, names_list  # Return all three

def get_person_name():
    root = Tk()
    root.withdraw()
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    root.destroy()
    return name

def is_point_in_polygon(point, polygon):
    """Return True if point is inside polygon, safely handling empty polygons."""
    if not polygon or len(polygon) < 3:  # Minimum 3 points for a polygon
        return False
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0

def draw_roi(frame):
    roi_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(roi_points) > 1:
                cv2.line(frame, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
            cv2.imshow("Draw ROI", frame)
        
    cv2.imshow("Draw ROI", frame)
    cv2.setMouseCallback("Draw ROI", mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(roi_points) > 2:
            cv2.line(frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
            break
    cv2.destroyWindow("Draw ROI")
    return roi_points

def initialize_queues():
    global frame_queues
    if not frame_queues:
        print("Initializing frame queues...")
        frame_queues.extend([mp.Queue(maxsize=2000) for _ in range(len(camera_urls))])
        # print(f"Initialized {len(frame_queues)} frame queues.")
        logger.info(f"Initialized {len(frame_queues)} frame queues.")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    requests.post(url, data=payload)

def send_photo_from_frame(frame, caption=None):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Lower quality for speed
    photo_bytes = BytesIO(buffer.tobytes())
    files = {"photo": ("alert.jpg", photo_bytes, "image/jpeg")}
    payload = {"chat_id": chat_id, "caption": caption or ""}
    try:
        response = requests.post(url, data=payload, files=files, timeout=5)  # Add timeout
        if not response.ok:
            logger.error(f"Telegram send failed: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Telegram send error: {str(e)}")

def generate_offline_frame(camera_id, width=854, height=480):
    # Create a blank image
    offline_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add text indicating offline status
    text = f"Camera {camera_id} offline"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(offline_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return offline_frame

def update_active_cameras_periodically():
    """Update the cached list of active cameras, first immediately, then periodically."""
    global active_cameras_cache
    while True:
        temp_active_cameras = []
        for i, url in enumerate(camera_urls):
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                temp_active_cameras.append(i)
                cap.release()
        with cache_lock:
            active_cameras_cache = temp_active_cameras
        logger.info(f"Updated active cameras cache: {active_cameras_cache}")
        time.sleep(30)  # Update every 30 seconds

def frame_consumer(camera_id, frame_queue, latest_frames, latest_raw_frames, latest_detections):
    while True:
        try:
            # Expect tuple of (raw_frame, processed_frame, bboxes)
            raw_frame, processed_frame, bboxes = frame_queue.get(timeout=1)
            latest_frames[camera_id] = processed_frame
            latest_raw_frames[camera_id] = raw_frame
            latest_detections[camera_id] = bboxes
            # logger.debug(f"Updated latest raw, processed frames, and detections for Camera {camera_id}")
        except Empty:
            # logger.debug(f"Queue empty for Camera {camera_id}")
            if camera_id not in latest_frames:
                offline_frame = generate_offline_frame(camera_id)
                latest_frames[camera_id] = offline_frame
                latest_raw_frames[camera_id] = offline_frame
                latest_detections[camera_id] = []
            time.sleep(0.1)

def extract_ip_from_url(url):
    """Extract the IP address from an RTSP URL."""
    match = re.search(r'@(\d+\.\d+\.\d+\.\d+):', url)
    return match.group(1) if match else "Unknown IP"

def reload_database(shared_namespace, database):
    """Reload database embeddings and names into shared_namespace."""
    try:
        _, embeddings_list, names_list = load_database(shared_namespace)
        if embeddings_list:
            shared_namespace.database_embeddings = torch.tensor(np.stack(embeddings_list))
            shared_namespace.database_names = names_list
            logger.info(f"Reloaded database: {len(names_list)} embeddings")
        else:
            shared_namespace.database_embeddings = None
            shared_namespace.database_names = []
            logger.warning("No embeddings loaded during reload")
    except Exception as e:
        logger.error(f"Error reloading database: {str(e)}")

def initialize_video_writer(camera_id, frame_width, frame_height, fps=15):
    """Initialize a video writer for a camera with a timestamped filename in a date-based folder."""
    # Create date-based subdirectory (e.g., 29-04-2025)
    current_date = datetime.now().strftime("%d-%m-%Y")
    camera_dir = os.path.join(recordings_dir, f"camera_{camera_id}", current_date)
    os.makedirs(camera_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(camera_dir, f"recording_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use H.264 codec
    writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        logger.error(f"Failed to initialize video writer for Camera {camera_id} at {video_path}")
        return None
    logger.info(f"Initialized video writer for Camera {camera_id} at {video_path}")
    return writer

def upload_file_to_s3(file_path, bucket_name, s3_path):
    """
    Upload a file to S3 and return True if successful, False otherwise.
    
    Args:
        file_path (str): Local path to the file
        bucket_name (str): S3 bucket name
        s3_path (str): Destination path in S3 (e.g., camera_0/29-04-2025/recording_20250429_111816.mp4)
    
    Returns:
        bool: True if upload was successful, False otherwise
    """
    s3_client = boto3.client('s3')
    try:
        s3_logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_path}")
        s3_client.upload_file(file_path, bucket_name, s3_path)
        s3_logger.info(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_path}")
        return True
    except ClientError as e:
        s3_logger.error(f"Failed to upload {file_path} to S3: {str(e)}")
        return False
    except Exception as e:
        s3_logger.error(f"Unexpected error uploading {file_path}: {str(e)}")
        return False

def delete_local_file(file_path):
    """
    Delete a local file and log the result.
    
    Args:
        file_path (str): Path to the file to delete
    """
    try:
        os.remove(file_path)
        s3_logger.info(f"Deleted local file: {file_path}")
    except OSError as e:
        s3_logger.error(f"Failed to delete {file_path}: {str(e)}")

def scan_and_upload_recordings():
    """
    Scan the recordings directory, upload all .mp4 files to S3, and delete them locally if uploaded successfully.
    """
    if not os.path.exists(RECORDINGS_DIR):
        s3_logger.warning(f"Recordings directory {RECORDINGS_DIR} does not exist")
        return

    uploaded_files = 0
    deleted_files = 0
    failed_uploads = 0

    # Walk through the recordings directory
    for root, _, files in os.walk(RECORDINGS_DIR):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                # Create S3 path by removing the base recordings directory
                s3_path = os.path.relpath(file_path, RECORDINGS_DIR).replace(os.sep, '/')
                # Upload to S3
                if upload_file_to_s3(file_path, BUCKET_NAME, s3_path):
                    # Delete local file only if upload was successful
                    delete_local_file(file_path)
                    uploaded_files += 1
                    deleted_files += 1
                else:
                    failed_uploads += 1
                    s3_logger.warning(f"Skipping deletion of {file_path} due to upload failure")

    s3_logger.info(f"Upload cycle completed: {uploaded_files} files uploaded, {deleted_files} files deleted, {failed_uploads} upload failures")

def delete_old_s3_recordings(bucket_name, retention_period_seconds):
    """
    Delete S3 objects older than the specified retention period (in seconds).
    
    Args:
        bucket_name (str): S3 bucket name
        retention_period_seconds (int): Retention period in seconds (e.g., 48 hours)
    """
    s3_client = boto3.client('s3')
    try:
        # Get current time in UTC
        current_time = datetime.utcnow()
        retention_threshold = current_time - timedelta(seconds=retention_period_seconds)
        s3_logger.info(f"Checking for S3 objects older than {retention_threshold} in s3://{bucket_name}")

        # List objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        objects_to_delete = []

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < retention_threshold:
                        objects_to_delete.append({'Key': obj['Key']})

        # Delete objects if any are found
        if objects_to_delete:
            s3_logger.info(f"Found {len(objects_to_delete)} objects to delete in s3://{bucket_name}")
            # Delete objects in batches (max 1000 per request)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i + 1000]
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': batch}
                )
                for obj in batch:
                    s3_logger.info(f"Deleted s3://{bucket_name}/{obj['Key']}")
        else:
            s3_logger.info(f"No objects older than {retention_period_seconds // 3600} hours found in s3://{bucket_name}")

    except ClientError as e:
        s3_logger.error(f"Failed to delete old S3 objects: {str(e)}")
    except Exception as e:
        s3_logger.error(f"Unexpected error deleting old S3 objects: {str(e)}")

def run_s3_upload_cycle():
    """
    Run the S3 upload, delete local files, and delete old S3 recordings cycle every 12 hours.
    """
    s3_logger.info("Starting S3 upload thread")
    while True:
        s3_logger.info("Starting new upload cycle")
        scan_and_upload_recordings()
        s3_logger.info("Starting S3 cleanup cycle")
        delete_old_s3_recordings(BUCKET_NAME, RETENTION_PERIOD)
        s3_logger.info(f"Waiting {UPLOAD_INTERVAL // 3600} hours until next upload cycle")
        time.sleep(UPLOAD_INTERVAL)

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crops, shared_data, database):
    """
    Process crops from multiple cameras to capture all views (front, back, left, right) at once.
    """
    view_directions = shared_data['view_directions']  # ['Front', 'Back', 'Right', 'Left']
    if len(crops) != len(view_directions):
        logger.warning(f"Expected {len(view_directions)} crops, got {len(crops)}. Aborting.")
        return False
    
    # Capture all views at once
    for i, crop in enumerate(crops):
        view_direction = view_directions[i]
        print(f"Captured {view_direction} view.")
        shared_data['capture_images'].append(crop)

    # Add the person to the database with all captured views
    embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
    database[shared_data['current_name']] = embeddings
    shared_data['adding_new_person'].value = False
    shared_data['capture_images'][:] = []  # Clear the list
    shared_data['current_view'].value = 0  # Reset view index
    if 'new_person_track_id' in shared_data:
        del shared_data['new_person_track_id']
    return True

def process_dynamic_multi_camera(shared_data, database, latest_raw_frames, latest_detections, cameras, threshold=28):
    """
    Process dynamic capture from multiple cameras, collecting crops only from detected bounding boxes.
    Stops when the total number of captured crops reaches the threshold (default 28).
    """
    frame_count = 0
    max_frames = 200  # Fallback limit to prevent infinite loop
    while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
        crops = []
        for camera_id in cameras[:4]:  # Use the 4 selected cameras
            raw_frame = latest_raw_frames.get(camera_id)
            bboxes = latest_detections.get(camera_id, [])
            if raw_frame is None or not bboxes:
                continue
            
            # Use the most confident bounding box (assuming sorted by confidence in process_camera_feed)
            x1, y1, x2, y2 = bboxes[0]
            person_crop = raw_frame[y1:y2, x1:x2]
            crops.append(person_crop)
        
        frame_count += 1
        if frame_count % 5 == 0 and crops:  # Capture every 5th frame if detections exist
            for crop in crops:
                if len(shared_data['capture_images']) < threshold:  # Stop at threshold
                    shared_data['capture_images'].append(crop)
            logger.info(f"Dynamic capture: frame {frame_count}, captured {len(crops)} crops, total {len(shared_data['capture_images'])}/{threshold}")
        
        time.sleep(0.05)  # Simulate frame rate (adjust as needed)

    # Save to database when threshold is met or max frames reached with some crops
    if shared_data['capture_images']:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        logger.info(f"Dynamic capture completed: {len(shared_data['capture_images'])} crops saved for {shared_data['current_name']}")
    else:
        logger.warning(f"No crops captured for {shared_data['current_name']} after {max_frames} frames")
    
    shared_data['dynamic_adding_new_person'].value = False
    shared_data['capture_images'][:] = []
    logger.info(f"Dynamic capture finished, is_active set to False for {shared_data['current_name']}")
    return True

def process_standard_detections(detections, frame, roi_points, movement_trails, database, camera_id, shared_namespace):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = det['embedding']
        match, similarity = match_embedding(query_embedding, database, shared_namespace)
        label = match if match else "Unknown"

        if is_point_in_polygon(center, roi_points):
            # flabel = f"{label} ({similarity:.2f})"
            flabel = f"{label}"
            color = (0, 255, 0)
        else:
            # flabel = f"{label} ({similarity:.2f})"
            flabel = f"{label}"
            color = (0, 0, 255)

        cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # cv2.putText(frame, str(track_id), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)

        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 1)

        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]
            entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
            exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)
            if entered_roi or exited_roi:
                current_time = time.time()
                if track_id not in last_alert_time or (current_time - last_alert_time[track_id]) > 10:
                    direction = "entered" if entered_roi else "exited"
                    message_alert = f"📷 Camera {camera_id} ALERT: Person {label} {direction} the ROI."
                    telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
                    last_alert_time[track_id] = current_time

    return frame, movement_trails

# Main processing function
def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database, shared_namespace):
    # Enable OpenCV logging for FFmpeg
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)

    if not cap.isOpened():
        logger.error(f"Camera {camera_id} failed to open: {camera_url}. Check RTSP URL, FFmpeg support, or network.")
        offline_frame = generate_offline_frame(camera_id)
        frame_queue.put((offline_frame, offline_frame, []))
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    # Initialize video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = initialize_video_writer(camera_id, frame_width, frame_height)
    recording_start_time = time.time()
    # batch_duration = 30*60  # 30 minutes in seconds

    movement_trails = {}
    tracked_embeddings = {}
    reconnect_attempts = 0
    max_attempts = 5

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning(f"Camera {camera_id} failed to read frame, attempt {reconnect_attempts + 1}/{max_attempts}. POS_MSEC: {cap.get(cv2.CAP_PROP_POS_MSEC)}")
            reconnect_attempts += 1
            if reconnect_attempts >= max_attempts:
                logger.error(f"Camera {camera_id} offline after {max_attempts} attempts")
                offline_frame = generate_offline_frame(camera_id)
                frame_queue.put((offline_frame, offline_frame, []))
                if video_writer:
                    video_writer.release()
                break

            time.sleep(1)
            cap.release()
            if video_writer:
                video_writer.release()
                video_writer = None
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
            if not cap.isOpened():
                logger.warning(f"Camera {camera_id} reconnection failed")
                offline_frame = generate_offline_frame(camera_id)
                frame_queue.put((offline_frame, offline_frame, []))
            else:
                # Reinitialize video writer after reconnection
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = initialize_video_writer(camera_id, frame_width, frame_height)
                recording_start_time = time.time()
            continue
        reconnect_attempts = 0

        # Save raw frame before processing
        raw_frame = frame.copy()

        # Perform YOLO detection
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
        current_detections = []
        bboxes = []
        crops = []
        detection_info = []

        for result in results:
            if result.boxes is None:
                continue
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1
                crops.append(person_crop)
                detection_info.append({
                    'crop': person_crop,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'track_id': track_id
                })

        # Batch process embeddings
        if crops:
            try:
                image_tensors = torch.stack([transform(crop) for crop in crops]).to(device)
                with torch.no_grad():
                    with autocast(device_type='cuda'):
                        embeddings = reid_model(image_tensors)
                embeddings = embeddings.cpu().numpy()
                for idx, (embedding, det_info) in enumerate(zip(embeddings, detection_info)):
                    tracked_embeddings[det_info['track_id']] = embedding
                    current_detections.append({
                        'crop': det_info['crop'],
                        'bbox': det_info['bbox'],
                        'center': det_info['center'],
                        'track_id': det_info['track_id'],
                        'embedding': embedding
                    })
            except Exception as e:
                logger.error(f"Error in batch embedding extraction: {str(e)}")
                for det_info in detection_info:
                    embedding = np.zeros(1000)
                    tracked_embeddings[det_info['track_id']] = embedding
                    current_detections.append({
                        'crop': det_info['crop'],
                        'bbox': det_info['bbox'],
                        'center': det_info['center'],
                        'track_id': det_info['track_id'],
                        'embedding': embedding
                    })

        roi_points = shared_rois[camera_id]
        processed_frame = frame
        if current_detections:
            processed_frame, movement_trails = process_standard_detections(
                current_detections, processed_frame, roi_points, movement_trails, database, camera_id, shared_namespace
            )

        if roi_points and len(roi_points) >= 3:
            cv2.polylines(processed_frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

        # Check for database update event
        if shared_data['database_update_event'].is_set():
            logger.info(f"Camera {camera_id} received database update signal, reloading...")
            reload_database(shared_namespace, database)
            shared_data['database_update_event'].clear()  # Clear event after reload
            logger.info(f"Camera {camera_id} database reloaded")

        # Write frame to video file
        if video_writer and frame is not None and frame.size > 0:
            try:
                video_writer.write(frame)  # Write raw frame
            except Exception as e:
                logger.error(f"Error writing frame to video for Camera {camera_id}: {str(e)}")

        # Check if 30 minutes have passed to start a new recording
        current_time = time.time()
        if current_time - recording_start_time >= batch_duration:
            if video_writer:
                video_writer.release()
                logger.info(f"Closed video writer for Camera {camera_id} after 30 minutes")
            video_writer = initialize_video_writer(camera_id, frame_width, frame_height)
            recording_start_time = current_time

        # Store raw frame, processed frame, and bounding boxes
        if frame_queue.qsize() > 450:
            frame_queue.get()
        frame_queue.put((raw_frame, cv2.resize(processed_frame, (854, 480)), bboxes))

        if len(tracked_embeddings) > 100:
            oldest_track = min(tracked_embeddings.keys(), key=lambda k: tracked_embeddings[k].size)
            tracked_embeddings.pop(oldest_track)

    offline_frame = generate_offline_frame(camera_id)
    frame_queue.put((offline_frame, offline_frame, []))
    if video_writer:
        video_writer.release()
        logger.info(f"Released video writer for Camera {camera_id} on process end")
    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
# FLASK FUNCTIONS                       #
#########################################

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login requests."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            return redirect(url_for('index', page=1))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handle logout requests."""
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/', defaults={'page': 1})
@app.route('/<int:page>')
@login_required
def index(page):
    view_mode = request.args.get('view', 'separate')
    cameras_per_page = 2 if view_mode == 'separate' else 6
    total_cameras = len(camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages
    start_idx = (page - 1) * cameras_per_page
    end_idx = min(start_idx + cameras_per_page, total_cameras)
    camera_indices = list(range(start_idx, end_idx))
    
    # Create a dictionary mapping camera indices to their IPs
    camera_data = {i: {"ip": extract_ip_from_url(camera_urls[i])} for i in range(len(camera_urls))}
    
    persons = list(database.keys())
    camera_count = len(camera_urls)
    # Pass the cached active cameras to the template
    with cache_lock:
        active_cameras = active_cameras_cache[:]
    
    logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}, persons={persons}, camera_count={camera_count}, active_cameras={active_cameras}")
    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages,
                          view_mode=view_mode,
                          persons=persons,
                          camera_count=camera_count,
                          active_cameras=active_cameras,
                          camera_data=camera_data)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    def generate():
        while True:
            frame = latest_frames.get(camera_id, generate_offline_frame(camera_id))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_initial_frame/<int:camera_id>')
@login_required
def get_initial_frame(camera_id):
    if camera_id >= len(camera_urls) or camera_id < 0:
        return "Camera not found", 404
    cap = cv2.VideoCapture(camera_urls[camera_id], cv2.CAP_FFMPEG)
    success, frame = cap.read()
    cap.release()
    if not success:
        return "Failed to capture frame", 500
    ret, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/save_roi/<int:camera_id>', methods=['POST'])
@login_required
def save_roi(camera_id):
    if camera_id >= len(camera_urls) or camera_id < 0:
        return jsonify({"error": "Invalid camera ID"}), 404
    roi_data = request.json.get('roi', [])
    roi_points = [(int(x), int(y)) for x, y in roi_data]
    rois[camera_id] = roi_points  # Update shared list
    print(f"Saved ROI for Camera {camera_id}: {rois[camera_id]}")
    return jsonify({"status": "ROI saved successfully"}), 200

@app.route('/remove_roi/<int:camera_id>', methods=['POST'])
@login_required
def remove_roi(camera_id):
    if camera_id >= len(camera_urls) or camera_id < 0:
        return jsonify({"error": "Invalid camera ID"}), 404
    rois[camera_id] = []  # Clear the ROI points for this camera
    logger.info(f"ROI removed for Camera {camera_id}")
    return jsonify({"status": "ROI removed successfully"}), 200

@app.route('/add_person', methods=['POST'])
@login_required
def add_person():
    data = request.json
    type_ = data.get('type')
    name = data.get('name')
    cameras = data.get('cameras', [])
    disable_detection_check = data.get('disable_detection_check', False)  # From previous mod
    threshold = data.get('threshold', 28)  # New parameter, default to 28
    if not name or not type_:
        return jsonify({"message": "Name and type are required"}), 400
    
    if type_ == 'static':
        shared_data['adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        
        # Fetch raw frames and use stored bounding boxes
        crops = []
        for camera_id in cameras[:4]:
            raw_frame = latest_raw_frames.get(camera_id)
            bboxes = latest_detections.get(camera_id, [])
            if raw_frame is None:
                logger.warning(f"No raw frame available for camera {camera_id}")
                crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))
                continue
            
            if bboxes and not disable_detection_check:
                x1, y1, x2, y2 = bboxes[0]  # Use most confident bounding box
                person_crop = raw_frame[y1:y2, x1:x2]
                crops.append(person_crop)
            else:
                logger.warning(f"No detections for camera {camera_id}, using full frame")
                crops.append(raw_frame)
        
        success = process_new_person(crops, shared_data, database)
        if success:
            shared_data['database_update_event'].set()
            return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
        else:
            shared_data['adding_new_person'].value = False
            return jsonify({"message": "Failed to capture all views"}), 500
    
    elif type_ == 'dynamic':
        shared_data['dynamic_adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        
        def run_dynamic_capture():
            success = process_dynamic_multi_camera(shared_data, database, latest_raw_frames, latest_detections, cameras, threshold)
            if success:
                shared_data['database_update_event'].set()
            logger.info(f"Dynamic capture thread completed for {name}, success: {success}")
        
        threading.Thread(target=run_dynamic_capture, daemon=True).start()
        logger.info(f"Started dynamic capture thread for {name}")
        return '', 204
    
    return jsonify({"message": "Invalid addition type"}), 400

@app.route('/remove_person', methods=['POST'])
@login_required
def remove_person():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({"message": "Name is required"}), 400
    
    if name in database:
        del database[name]
        person_dir = os.path.join(database_dir, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        shared_data['database_update_event'].set()  # Signal update
        return jsonify({"message": f"Removed {name} from database and deleted their images"}), 200
    return jsonify({"message": f"{name} not found in database"}), 404

@app.route('/clear_database', methods=['POST'])
@login_required
def clear_database():
    database.clear()
    # if os.path.exists(database_dir):
    #     shutil.rmtree(database_dir)
    # os.makedirs(database_dir, exist_ok=True)
    shared_data['database_update_event'].set()  # Signal update
    return jsonify({"message": "Database cleared successfully"}), 200

@app.route('/get_person_list')
@login_required
def get_person_list():
    try:
        logger.info("Received request for /get_person_list")
        persons = list(database.keys())
        logger.info(f"Returning person list: {persons}")
        return jsonify({"persons": persons})
    except Exception as e:
        logger.error(f"Error in get_person_list: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_active_cameras')
@login_required
def get_active_cameras():
    """Return the cached list of active cameras (optional, kept for debugging)."""
    with cache_lock:
        return jsonify({"active_cameras": active_cameras_cache})

@app.route('/capture_static', methods=['GET'])
@login_required
def capture_static():
    name = request.args.get('name')
    cameras = request.args.get('cameras')  # Comma-separated list of camera indices
    if not name or not cameras:
        flash("Name and cameras are required for capture.")
        return redirect(url_for('index'))
    
    try:
        camera_indices = [int(i) for i in cameras.split(',')][:4]  # Limit to 4 cameras
        if not all(0 <= i < len(camera_urls) for i in camera_indices):
            raise ValueError("Invalid camera indices")
    except ValueError:
        flash("Invalid camera selection.")
        return redirect(url_for('index'))
    
    # Create camera_data with IPs for the selected camera indices
    camera_data = {i: {"ip": extract_ip_from_url(camera_urls[i])} for i in camera_indices}
    
    return render_template('capture_static.html', 
                          name=name, 
                          camera_indices=camera_indices,
                          camera_data=camera_data)

@app.route('/capture_dynamic', methods=['GET'])
@login_required
def capture_dynamic():
    name = request.args.get('name')
    cameras = request.args.get('cameras')
    if not name or not cameras:
        flash("Name and cameras are required for capture.")
        return redirect(url_for('index'))
    
    try:
        camera_indices = [int(i) for i in cameras.split(',')][:4]  # Limit to 4 cameras
        if not all(0 <= i < len(camera_urls) for i in camera_indices):
            raise ValueError("Invalid camera indices")
    except ValueError:
        flash("Invalid camera selection.")
        return redirect(url_for('index'))
    
    camera_data = {i: {"ip": extract_ip_from_url(camera_urls[i])} for i in camera_indices}
    
    return render_template('capture_dynamic.html', 
                          name=name, 
                          camera_indices=camera_indices,
                          camera_data=camera_data)

@app.route('/check_dynamic_status', methods=['GET'])
@login_required
def check_dynamic_status():
    is_active = shared_data['dynamic_adding_new_person'].value
    return jsonify({"is_active": is_active})

@app.route('/get_dynamic_crop_count', methods=['GET'])
@login_required
def get_dynamic_crop_count():
    crop_count = len(shared_data['capture_images'])
    is_active = shared_data['dynamic_adding_new_person'].value
    return jsonify({"crop_count": crop_count, "is_active": is_active})

@app.route('/check_person_and_redirect', methods=['POST'])
@login_required
def check_person_and_redirect():
    data = request.json
    name = data.get('name')
    cameras = data.get('cameras', [])
    
    if not name or not cameras:
        return jsonify({"message": "Name and cameras are required"}), 400
    
    if name not in database:
        return jsonify({"message": "Person not found in the database, use static addition method for new staff."}), 404
    
    # Name exists, proceed to dynamic capture
    camera_param = ','.join(map(str, cameras[:4]))  # Limit to 4 cameras
    return jsonify({"redirect": f"/capture_dynamic?name={urllib.parse.quote(name)}&cameras={camera_param}"}), 200

@app.route('/refresh_cameras', methods=['POST'])
@login_required
def refresh_cameras():
    global processes
    logger.info("Refreshing camera processes...")
    restarted = 0
    for i in range(len(camera_urls)):
        p = processes[i]
        if not p.is_alive():
            logger.info(f"Restarting process for Camera {i}...")
            p.terminate()
            p.join()  # Ensure the old process is fully terminated
            new_p = mp.Process(target=process_camera_feed,
                              args=(camera_urls[i], i, rois, frame_queues[i], shared_data, database, shared_namespace),
                              daemon=True)
            new_p.start()
            processes[i] = new_p
            restarted += 1
            logger.info(f"Restarted process for Camera {i}: {camera_urls[i]}")
    return jsonify({"message": f"Refreshed camera streams. Restarted {restarted} processes."}), 200

# @app.route('/recordings')
# @login_required
# def recordings():
#     """List all .mp4 recordings in the S3 bucket and render the recordings page."""
#     s3_client = boto3.client('s3')
#     recordings = []
#     try:
#         paginator = s3_client.get_paginator('list_objects_v2')
#         pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix='')
#         for page in pages:
#             if 'Contents' in page:
#                 for obj in page['Contents']:
#                     key = obj['Key']
#                     if key.endswith('.mp4'):
#                         # Extract metadata from key (e.g., camera_0/29-04-2025/recording_20250429_111816.mp4)
#                         parts = key.split('/')
#                         if len(parts) == 3:
#                             camera_part, date_part, file_name = parts
#                             camera = camera_part.replace('camera_', '')
#                             date = date_part
#                             timestamp = file_name.replace('recording_', '').replace('.mp4', '')
#                             timestamp = f"{timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
#                             size_mb = round(obj['Size'] / (1024 * 1024), 2)  # Convert bytes to MB
#                             recordings.append({
#                                 'key': key,
#                                 'file_name': file_name,
#                                 'camera': camera,
#                                 'date': date,
#                                 'timestamp': timestamp,
#                                 'size_mb': size_mb
#                             })
#     except ClientError as e:
#         logger.error(f"Failed to list S3 objects: {str(e)}")
#         flash('Unable to retrieve recordings due to an S3 error.')
#     except Exception as e:
#         logger.error(f"Unexpected error listing S3 objects: {str(e)}")
#         flash('An unexpected error occurred while retrieving recordings.')

#     return render_template('recordings.html', recordings=recordings)



@app.route('/recordings')
@login_required
def recordings():
    """List all .mp4 recordings in the S3 bucket in a hierarchical structure."""
    s3_client = boto3.client('s3')
    recordings_tree = {}
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix='')
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.mp4'):
                        # Extract metadata from key (e.g., camera_0/29-04-2025/recording_20250429_111816.mp4)
                        parts = key.split('/')
                        if len(parts) == 3:
                            camera_part, date_part, file_name = parts
                            camera = camera_part  # Keep 'camera_0' as key
                            date = date_part
                            timestamp = file_name.replace('recording_', '').replace('.mp4', '')
                            timestamp = f"{timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
                            size_mb = round(obj['Size'] / (1024 * 1024), 2)  # Convert bytes to MB
                            
                            # Initialize nested dictionary
                            if camera not in recordings_tree:
                                recordings_tree[camera] = {}
                            if date not in recordings_tree[camera]:
                                recordings_tree[camera][date] = []
                            
                            # Add recording to the appropriate camera and date
                            recordings_tree[camera][date].append({
                                'key': key,
                                'file_name': file_name,
                                'timestamp': timestamp,
                                'size_mb': size_mb
                            })
    except ClientError as e:
        logger.error(f"Failed to list S3 objects: {str(e)}")
        flash('Unable to retrieve recordings due to an S3 error.')
    except Exception as e:
        logger.error(f"Unexpected error listing S3 objects: {str(e)}")
        flash('An unexpected error occurred while retrieving recordings.')

    return render_template('recordings.html', recordings_tree=recordings_tree)


@app.route('/download_recordings', methods=['POST'])
@login_required
def download_recordings():
    """Create a ZIP file of selected recordings and return a presigned URL for download."""
    data = request.get_json()
    keys = data.get('keys', [])
    if not keys:
        return jsonify({"error": "No recordings selected"}), 400

    s3_client = boto3.client('s3')
    temp_dir = None
    zip_path = None
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_filename = f"recordings_{timestamp}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)

        # Download selected files from S3 to temporary directory
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for key in keys:
                file_name = key.split('/')[-1]  # e.g., recording_20250429_111816.mp4
                local_path = os.path.join(temp_dir, file_name)
                try:
                    s3_client.download_file(BUCKET_NAME, key, local_path)
                    zipf.write(local_path, file_name)  # Add file to ZIP
                    s3_logger.info(f"Added {file_name} to ZIP")
                except ClientError as e:
                    s3_logger.error(f"Failed to download {key} from S3: {str(e)}")
                    return jsonify({"error": f"Failed to download {file_name} from S3"}), 500

        # Upload ZIP file to S3
        s3_zip_key = f"temp_zips/{zip_filename}"
        s3_client.upload_file(zip_path, BUCKET_NAME, s3_zip_key)
        s3_logger.info(f"Uploaded ZIP file to s3://{BUCKET_NAME}/{s3_zip_key}")

        # Generate presigned URL for the ZIP file
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_zip_key},
            ExpiresIn=3600  # 1 hour expiration
        )

        return jsonify({"url": url, "filename": zip_filename}), 200

    except ClientError as e:
        logger.error(f"Failed to process ZIP file: {str(e)}")
        return jsonify({"error": "Failed to create or upload ZIP file due to an S3 error"}), 500
    except Exception as e:
        logger.error(f"Unexpected error processing ZIP file: {str(e)}")
        return jsonify({"error": "An unexpected error occurred while creating the ZIP file"}), 500
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                s3_logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                s3_logger.error(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")


#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    # Create recordings directory
    os.makedirs(recordings_dir, exist_ok=True)

    manager = mp.Manager()
    shared_namespace = manager.Namespace()
    shared_namespace.database_embeddings = None
    shared_namespace.database_names = []
    database_dict, embeddings_list, names_list = load_database(shared_namespace)
    database = manager.dict(database_dict)
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"],
        'selected_cameras': manager.list(),
        'database_update_event': manager.Event()
    })

    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()
    latest_raw_frames = manager.dict()  # Initialize raw frames
    latest_detections = manager.dict()  # Initialize detections

    cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
    cache_thread.start()

    # Start S3 upload thread
    s3_upload_thread = threading.Thread(target=run_s3_upload_cycle, daemon=True)
    s3_upload_thread.start()

    initialize_queues()

    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database, shared_namespace),
                       daemon=True)
        processes.append(p)
        p.start()
        # logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames, latest_raw_frames, latest_detections),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        # logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    server_thread.join()































