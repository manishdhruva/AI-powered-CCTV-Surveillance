
#########################################
#   INSTALL AND SET DEPENDENCIES        #
#########################################

import re
import os
import cv2
import time
import torch
import shutil
import logging
import requests
import torchreid
import threading
import numpy as np
import urllib.parse
from io import BytesIO
from functools import wraps
from ultralytics import YOLO
from torchvision import transforms
import torch.multiprocessing as mp
from torch.amp import autocast
from tkinter import Tk, simpledialog
from multiprocessing.queues import Empty
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Simulated user database
USERS = {"admin": "admin"}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
try:
    body_model = YOLO("yolo11n_12-2.engine", task='detect')
    # logger.info("Body YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load body YOLO model: {str(e)}")
    raise

try:
    face_model = YOLO("yolov11n-face_12-2.engine", task='detect')
    # logger.info("Face YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load face YOLO model: {str(e)}")
    raise

torch.cuda.empty_cache()
# Initialize ReID model for body
reid_model = torchreid.models.build_model(
    # name='osnet_ibn_x1_0',
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
).to(device)
reid_model.eval()

# Preprocessing for ReID
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()
])

# Preprocessing for face images
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),  # Adjust based on face recognition model
    transforms.ToTensor(),
])

telegram_pool = ThreadPoolExecutor(max_workers=5)

#########################################
#   GLOBAL VARIABLES                    #
#########################################

database_dir = './body_database'
face_database_dir = './face_database'

# Global dictionaries
latest_frames = {}
latest_raw_frames = {}
latest_detections = {}
frame_queues = []

# Camera URLs
camera_urls = {
    'body-cameras': [
        "rtsp://admin:Lucky786$1@192.168.1.121:554/Streaming/Channels/101?transport=tcp",
        "rtsp://admin:Lucky786$1@192.168.1.137:554/Streaming/Channels/101?transport=tcp",
        "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp",
        "rtsp://admin:Lucky786$1@192.168.1.117:554/Streaming/Channels/101?transport=tcp",
        "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp",
    ],
    'face-cameras': [
        "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp",
    ]
}

# Flatten camera URLs for indexing
all_camera_urls = camera_urls['body-cameras'] + camera_urls['face-cameras']
camera_types = {i: 'body' for i in range(len(camera_urls['body-cameras']))}
camera_types.update({i + len(camera_urls['body-cameras']): 'face' for i in range(len(camera_urls['face-cameras']))})

# Telegram credentials
bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
chat_id = '-1002599945824'

# Cache for active cameras
active_cameras_cache = list(range(len(all_camera_urls)))
cache_lock = threading.Lock()

last_alert_time = {}
body_database_embeddings = None
body_database_names = []
face_database = {}

# Process list
processes = []

##############################
#   HELPER FUNCTIONS         #
##############################

def add_body_to_database(name, images):
    person_dir = os.path.join(database_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    embeddings = []
    for idx, image in enumerate(images):
        image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
        cv2.imwrite(image_filename, image)
        embeddings.append(extract_body_embedding(image))
    return embeddings

def add_face_to_database(name, images):
    person_dir = os.path.join(face_database_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    for idx, image in enumerate(images):
        image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
        cv2.imwrite(image_filename, image)
    return True

def extract_body_embedding(image):
    try:
        if image.size == 0:
            logger.error("Empty image for body embedding")
            return np.zeros(1000)
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast(device_type='cuda'):
                embedding = reid_model(image_tensor)
        embedding = embedding.squeeze(0).cpu().numpy()
        return embedding
    except Exception as e:
        logger.error(f"Error in extract_body_embedding: {str(e)}")
        return np.zeros(1000)

def extract_face_embedding(image):
    try:
        if image is None or image.size == 0:
            logger.error("Empty or invalid image for face embedding")
            return np.zeros(1000)  # Match ReID embedding size
        # Use the same preprocessing as body embedding
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast(device_type='cuda'):
                embedding = reid_model(image_tensor)
        embedding = embedding.squeeze(0).cpu().numpy()
        return embedding
    except Exception as e:
        logger.error(f"Error in extract_face_embedding: {str(e)}")
        return np.zeros(1000)

def match_body_embedding(query_embedding, database, shared_namespace, threshold=0.7):
    if shared_namespace.body_database_embeddings is None or len(shared_namespace.body_database_embeddings) == 0:
        return None, 0
    try:
        db_embeddings = shared_namespace.body_database_embeddings.to(device)
        query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0).to(device)
        with torch.no_grad():
            similarities = torch.cosine_similarity(query_tensor, db_embeddings)
            best_similarity, best_idx = torch.max(similarities, dim=0)
            best_similarity = best_similarity.item()
            if best_similarity >= threshold:
                return shared_namespace.body_database_names[best_idx], best_similarity
        return None, 0
    except Exception as e:
        logger.error(f"Error in match_body_embedding: {str(e)}")
        return None, 0

def match_face_embedding(query_embedding, database, threshold=0.7):  # Adjust threshold if needed
    if not database:
        return None, 0
    try:
        max_similarity = 0
        best_match = None
        for name, embeddings in database.items():
            for emb in embeddings:
                similarity = cosine_similarity([query_embedding], [emb])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = name
        if max_similarity >= threshold:
            return best_match, max_similarity
        return None, 0
    except Exception as e:
        logger.error(f"Error in match_face_embedding: {str(e)}")
        return None, 0

def load_body_database(shared_namespace):
    database = {}
    embeddings_list = []
    names_list = []
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        embedding = extract_body_embedding(image)
                        embeddings.append(embedding)
                        embeddings_list.append(embedding)
                        names_list.append(person_name)
                database[person_name] = embeddings
        if embeddings_list:
            shared_namespace.body_database_embeddings = torch.tensor(np.stack(embeddings_list))
            shared_namespace.body_database_names = names_list
    return database, embeddings_list, names_list

def load_face_database():
    database = {}
    if os.path.exists(face_database_dir):
        for person_name in os.listdir(face_database_dir):
            person_dir = os.path.join(face_database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_filename)
                    image = cv2.imread(image_path)
                    embedding = extract_face_embedding(image)
                    if embedding is not None:
                        embeddings.append(embedding)
                if embeddings:
                    database[person_name] = embeddings
    return database

def get_person_name():
    root = Tk()
    root.withdraw()
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    root.destroy()
    return name

def is_point_in_polygon(point, polygon):
    if not polygon or len(polygon) < 3:
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
    frame_queues.extend([mp.Queue(maxsize=2000) for _ in range(len(all_camera_urls))])
    logger.info(f"Initialized {len(frame_queues)} frame queues.")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(url, data=payload)

def send_photo_from_frame(frame, caption=None):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    photo_bytes = BytesIO(buffer.tobytes())
    files = {"photo": ("alert.jpg", photo_bytes, "image/jpeg")}
    payload = {"chat_id": chat_id, "caption": caption or ""}
    try:
        response = requests.post(url, data=payload, files=files, timeout=5)
        if not response.ok:
            logger.error(f"Telegram send failed: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Telegram send error: {str(e)}")

def generate_offline_frame(camera_id, width=854, height=480):
    offline_frame = np.zeros((height, width, 3), dtype=np.uint8)
    text = f"Camera {camera_id} ({camera_types[camera_id]}) offline"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(offline_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return offline_frame

def update_active_cameras_periodically():
    global active_cameras_cache
    while True:
        temp_active_cameras = []
        for i, url in enumerate(all_camera_urls):
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                temp_active_cameras.append(i)
                cap.release()
        with cache_lock:
            active_cameras_cache = temp_active_cameras
        logger.info(f"Updated active cameras cache: {active_cameras_cache}")
        time.sleep(30)

def frame_consumer(camera_id, frame_queue, latest_frames, latest_raw_frames, latest_detections):
    while True:
        try:
            raw_frame, processed_frame, bboxes = frame_queue.get(timeout=1)
            latest_frames[camera_id] = processed_frame
            latest_raw_frames[camera_id] = raw_frame
            latest_detections[camera_id] = bboxes
        except Empty:
            if camera_id not in latest_frames:
                offline_frame = generate_offline_frame(camera_id)
                latest_frames[camera_id] = offline_frame
                latest_raw_frames[camera_id] = offline_frame
                latest_detections[camera_id] = []
            time.sleep(0.1)

def extract_ip_from_url(url):
    match = re.search(r'@(\d+\.\d+\.\d+\.\d+):', url)
    return match.group(1) if match else "Unknown IP"

def reload_databases(shared_namespace, body_database, face_database):
    _, embeddings_list, names_list = load_body_database(shared_namespace)
    if embeddings_list:
        shared_namespace.body_database_embeddings = torch.tensor(np.stack(embeddings_list))
        shared_namespace.body_database_names = names_list
        logger.info(f"Reloaded body database: {len(names_list)} embeddings")
    else:
        shared_namespace.body_database_embeddings = None
        shared_namespace.body_database_names = []
    face_database.clear()
    face_database.update(load_face_database())
    logger.info(f"Reloaded face database: {len(face_database)} persons")

#########################################
# Processing Functions                  #
#########################################

def process_new_body_person(crops, shared_data, database):
    view_directions = shared_data['view_directions']
    if len(crops) != len(view_directions):
        logger.warning(f"Expected {len(view_directions)} crops, got {len(crops)}")
        return False
    for i, crop in enumerate(crops):
        shared_data['capture_images'].append(crop)
    embeddings = add_body_to_database(shared_data['current_name'], shared_data['capture_images'])
    database[shared_data['current_name']] = embeddings
    shared_data['adding_new_person'].value = False
    shared_data['capture_images'][:] = []
    shared_data['current_view'].value = 0
    return True

def process_new_face_person(crops, shared_data, database):
    if not crops:
        logger.warning("No face crops provided")
        return False
    success = add_face_to_database(shared_data['current_name'], crops)
    shared_data['adding_new_person'].value = False
    shared_data['capture_images'][:] = []
    return success

def process_dynamic_multi_camera(shared_data, database, latest_raw_frames, latest_detections, cameras, threshold=28, camera_type='body'):
    frame_count = 0
    max_frames = 200
    while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
        crops = []
        for camera_id in cameras[:4]:
            raw_frame = latest_raw_frames.get(camera_id)
            bboxes = latest_detections.get(camera_id, [])
            if raw_frame is None or not bboxes:
                continue
            x1, y1, x2, y2 = bboxes[0]
            crop = raw_frame[y1:y2, x1:x2]
            crops.append(crop)
        frame_count += 1
        if frame_count % 5 == 0 and crops:
            for crop in crops:
                if len(shared_data['capture_images']) < threshold:
                    shared_data['capture_images'].append(crop)
        time.sleep(0.05)
    if shared_data['capture_images']:
        if camera_type == 'body':
            embeddings = add_body_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
        else:
            add_face_to_database(shared_data['current_name'], shared_data['capture_images'])
    shared_data['dynamic_adding_new_person'].value = False
    shared_data['capture_images'][:] = []
    return True

def process_body_detections(detections, frame, roi_points, movement_trails, database, camera_id, shared_namespace):
    if not detections:
        return frame, movement_trails
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = det['embedding']
        match, similarity = match_body_embedding(query_embedding, database, shared_namespace)
        label = match if match else "Unknown"
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label}"
            color = (0, 255, 0)
        else:
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
                    message_alert = f"📷 Camera {camera_id} (Body) ALERT: Person {label} {direction} the ROI."
                    telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
                    last_alert_time[track_id] = current_time
    return frame, movement_trails

def process_face_detections(detections, frame, roi_points, movement_trails, database, camera_id):
    if not detections:
        return frame, movement_trails
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = det['embedding']
        match, similarity = match_face_embedding(query_embedding, database)
        label = match if match else "Unknown"
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label}"
            color = (0, 255, 0)
        else:
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
        # for j in range(1, len(movement_trails[track_id])):
        #     cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 1)
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]
            entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
            exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)
            if entered_roi or exited_roi:
                current_time = time.time()
                if track_id not in last_alert_time or (current_time - last_alert_time[track_id]) > 10:
                    direction = "entered" if entered_roi else "exited"
                    message_alert = f"📷 Camera {camera_id} (Face) ALERT: Person {label} {direction} the ROI."
                    telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
                    last_alert_time[track_id] = current_time
    return frame, movement_trails

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, body_database, face_database, shared_namespace):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
    if not cap.isOpened():
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        offline_frame = generate_offline_frame(camera_id)
        frame_queue.put((offline_frame, offline_frame, []))
        return
    logger.info(f"Camera {camera_id} ({camera_types[camera_id]}) initialized: {camera_url}")
    movement_trails = {}
    tracked_embeddings = {}
    reconnect_attempts = 0
    max_attempts = 5
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning(f"Camera {camera_id} failed to read frame, attempt {reconnect_attempts + 1}/{max_attempts}")
            reconnect_attempts += 1
            if reconnect_attempts >= max_attempts:
                logger.error(f"Camera {camera_id} offline after {max_attempts} attempts")
                offline_frame = generate_offline_frame(camera_id)
                frame_queue.put((offline_frame, offline_frame, []))
                break
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
            if not cap.isOpened():
                offline_frame = generate_offline_frame(camera_id)
                frame_queue.put((offline_frame, offline_frame, []))
            continue
        reconnect_attempts = 0
        raw_frame = frame.copy()
        current_detections = []
        bboxes = []
        camera_type = camera_types[camera_id]
        if camera_type == 'body':
            results = body_model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
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
            # Batch process body embeddings
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
                    logger.error(f"Error in batch body embedding extraction: {str(e)}")
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
        else:
            results = face_model.track(frame, stream=False, persist=True, device=device, conf=0.5)
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
                    face_crop = frame[y1:y2, x1:x2]
                    track_id = int(track_id) if track_id is not None else -1
                    crops.append(face_crop)
                    detection_info.append({
                        'crop': face_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
            # Batch process face embeddings
            if crops:
                try:
                    image_tensors = torch.stack([transform(crop) for crop in crops]).to(device)
                    with torch.no_grad():
                        with autocast(device_type='cuda'):
                            embeddings = reid_model(image_tensors)
                    embeddings = embeddings.cpu().numpy()
                    for idx, (embedding, det_info) in enumerate(zip(embeddings, detection_info)):
                        if embedding is not None:
                            tracked_embeddings[det_info['track_id']] = embedding
                            current_detections.append({
                                'crop': det_info['crop'],
                                'bbox': det_info['bbox'],
                                'center': det_info['center'],
                                'track_id': det_info['track_id'],
                                'embedding': embedding
                            })
                except Exception as e:
                    logger.error(f"Error in batch face embedding extraction: {str(e)}")
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
            if camera_type == 'body':
                processed_frame, movement_trails = process_body_detections(
                    current_detections, processed_frame, roi_points, movement_trails, body_database, camera_id, shared_namespace
                )
            else:
                processed_frame, movement_trails = process_face_detections(
                    current_detections, processed_frame, roi_points, movement_trails, face_database, camera_id
                )
        if roi_points and len(roi_points) >= 3:
            cv2.polylines(processed_frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        if shared_data['database_update_event'].is_set():
            logger.info(f"Camera {camera_id} reloading databases")
            reload_databases(shared_namespace, body_database, face_database)
            shared_data['database_update_event'].clear()
        if frame_queue.qsize() > 450:
            frame_queue.get()
        frame_queue.put((raw_frame, cv2.resize(processed_frame, (854, 480)), bboxes))
        if len(tracked_embeddings) > 100:
            oldest_track = min(tracked_embeddings.keys(), key=lambda k: tracked_embeddings[k].size)
            tracked_embeddings.pop(oldest_track)
    offline_frame = generate_offline_frame(camera_id)
    frame_queue.put((offline_frame, offline_frame, []))
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
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/', defaults={'page': 1})
@app.route('/<int:page>')
@login_required
def index(page):
    view_mode = request.args.get('view', 'separate')
    cameras_per_page = 2 if view_mode == 'separate' else 6
    total_cameras = len(all_camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages
    start_idx = (page - 1) * cameras_per_page
    end_idx = min(start_idx + cameras_per_page, total_cameras)
    camera_indices = list(range(start_idx, end_idx))
    camera_data = {i: {"ip": extract_ip_from_url(all_camera_urls[i]), "type": camera_types[i]} for i in range(len(all_camera_urls))}
    persons = list(set(list(body_database.keys()) + list(face_database.keys())))
    camera_count = len(all_camera_urls)
    with cache_lock:
        active_cameras = active_cameras_cache[:]
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
    if camera_id >= len(all_camera_urls) or camera_id < 0:
        return "Camera not found", 404
    cap = cv2.VideoCapture(all_camera_urls[camera_id], cv2.CAP_FFMPEG)
    success, frame = cap.read()
    cap.release()
    if not success:
        return "Failed to capture frame", 500
    ret, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/save_roi/<int:camera_id>', methods=['POST'])
@login_required
def save_roi(camera_id):
    if camera_id >= len(all_camera_urls) or camera_id < 0:
        return jsonify({"error": "Invalid camera ID"}), 404
    roi_data = request.json.get('roi', [])
    roi_points = [(int(x), int(y)) for x, y in roi_data]
    rois[camera_id] = roi_points
    return jsonify({"status": "ROI saved successfully"}), 200

@app.route('/remove_roi/<int:camera_id>', methods=['POST'])
@login_required
def remove_roi(camera_id):
    if camera_id >= len(all_camera_urls) or camera_id < 0:
        return jsonify({"error": "Invalid camera ID"}), 404
    rois[camera_id] = []
    return jsonify({"status": "ROI removed successfully"}), 200

@app.route('/add_person', methods=['POST'])
@login_required
def add_person():
    data = request.json
    type_ = data.get('type')
    name = data.get('name')
    cameras = data.get('cameras', [])
    disable_detection_check = data.get('disable_detection_check', False)
    threshold = data.get('threshold', 28)
    detection_type = data.get('detection_type', 'body')  # 'body' or 'face'
    if not name or not type_ or not detection_type:
        return jsonify({"message": "Name, type, and detection_type are required"}), 400
    if type_ == 'static':
        shared_data['adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        crops = []
        for camera_id in cameras[:4]:
            raw_frame = latest_raw_frames.get(camera_id)
            bboxes = latest_detections.get(camera_id, [])
            if raw_frame is None:
                crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))
                continue
            if bboxes and not disable_detection_check:
                x1, y1, x2, y2 = bboxes[0]
                crop = raw_frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(raw_frame)
        success = False
        if detection_type == 'body':
            success = process_new_body_person(crops, shared_data, body_database)
        else:
            success = process_new_face_person(crops, shared_data, face_database)
        if success:
            shared_data['database_update_event'].set()
            return jsonify({"message": f"Captured for {name} ({detection_type}) and added to database"}), 200
        else:
            shared_data['adding_new_person'].value = False
            return jsonify({"message": "Failed to capture"}), 500
    elif type_ == 'dynamic':
        shared_data['dynamic_adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        def run_dynamic_capture():
            success = process_dynamic_multi_camera(shared_data, body_database if detection_type == 'body' else face_database,
                                                  latest_raw_frames, latest_detections, cameras, threshold, detection_type)
            if success:
                shared_data['database_update_event'].set()
        threading.Thread(target=run_dynamic_capture, daemon=True).start()
        return '', 204
    return jsonify({"message": "Invalid addition type"}), 400

@app.route('/remove_person', methods=['POST'])
@login_required
def remove_person():
    data = request.json
    name = data.get('name')
    detection_type = data.get('detection_type', 'body')
    if not name:
        return jsonify({"message": "Name is required"}), 400
    success = False
    if detection_type == 'body' and name in body_database:
        del body_database[name]
        person_dir = os.path.join(database_dir, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        success = True
    elif detection_type == 'face' and name in face_database:
        del face_database[name]
        person_dir = os.path.join(face_database_dir, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        success = True
    if success:
        shared_data['database_update_event'].set()
        return jsonify({"message": f"Removed {name} from {detection_type} database"}), 200
    return jsonify({"message": f"{name} not found in {detection_type} database"}), 404

@app.route('/clear_database', methods=['POST'])
@login_required
def clear_database():
    detection_type = request.json.get('detection_type', 'body')
    if detection_type == 'body':
        body_database.clear()
        if os.path.exists(database_dir):
            shutil.rmtree(database_dir)
        os.makedirs(database_dir, exist_ok=True)
    else:
        face_database.clear()
        if os.path.exists(face_database_dir):
            shutil.rmtree(face_database_dir)
        os.makedirs(face_database_dir, exist_ok=True)
    shared_data['database_update_event'].set()
    return jsonify({"message": f"{detection_type} database cleared successfully"}), 200

@app.route('/get_person_list')
@login_required
def get_person_list():
    detection_type = request.args.get('detection_type', 'body')
    persons = list(body_database.keys()) if detection_type == 'body' else list(face_database.keys())
    return jsonify({"persons": persons})

@app.route('/get_active_cameras')
@login_required
def get_active_cameras():
    with cache_lock:
        return jsonify({"active_cameras": active_cameras_cache})

@app.route('/capture_static', methods=['GET'])
@login_required
def capture_static():
    name = request.args.get('name')
    cameras = request.args.get('cameras')
    detection_type = request.args.get('detection_type', 'body')
    if not name or not cameras:
        flash("Name and cameras are required for capture.")
        return redirect(url_for('index'))
    try:
        camera_indices = [int(i) for i in cameras.split(',')][:4]
        if not all(0 <= i < len(all_camera_urls) for i in camera_indices):
            raise ValueError("Invalid camera indices")
    except ValueError:
        flash("Invalid camera selection.")
        return redirect(url_for('index'))
    camera_data = {i: {"ip": extract_ip_from_url(all_camera_urls[i]), "type": camera_types[i]} for i in camera_indices}
    return render_template('capture_static.html',
                          name=name,
                          camera_indices=camera_indices,
                          camera_data=camera_data,
                          detection_type=detection_type)

@app.route('/capture_dynamic', methods=['GET'])
@login_required
def capture_dynamic():
    name = request.args.get('name')
    cameras = request.args.get('cameras')
    detection_type = request.args.get('detection_type', 'body')
    if not name or not cameras:
        flash("Name and cameras are required for capture.")
        return redirect(url_for('index'))
    try:
        camera_indices = [int(i) for i in cameras.split(',')][:4]
        if not all(0 <= i < len(all_camera_urls) for i in camera_indices):
            raise ValueError("Invalid camera indices")
    except ValueError:
        flash("Invalid camera selection.")
        return redirect(url_for('index'))
    camera_data = {i: {"ip": extract_ip_from_url(all_camera_urls[i]), "type": camera_types[i]} for i in camera_indices}
    return render_template('capture_dynamic.html',
                          name=name,
                          camera_indices=camera_indices,
                          camera_data=camera_data,
                          detection_type=detection_type)

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
    detection_type = data.get('detection_type', 'body')
    if not name or not cameras:
        return jsonify({"message": "Name and cameras are required"}), 400
    database_to_check = body_database if detection_type == 'body' else face_database
    if name not in database_to_check:
        return jsonify({"message": f"Person not found in {detection_type} database, use static addition method."}), 404
    camera_param = ','.join(map(str, cameras[:4]))
    return jsonify({"redirect": f"/capture_dynamic?name={urllib.parse.quote(name)}&cameras={camera_param}&detection_type={detection_type}"}), 200

@app.route('/refresh_cameras', methods=['POST'])
@login_required
def refresh_cameras():
    global processes
    restarted = 0
    for i in range(len(all_camera_urls)):
        p = processes[i]
        if not p.is_alive():
            p.terminate()
            p.join()
            new_p = mp.Process(target=process_camera_feed,
                              args=(all_camera_urls[i], i, rois, frame_queues[i], shared_data, body_database, face_database, shared_namespace),
                              daemon=True)
            new_p.start()
            processes[i] = new_p
            restarted += 1
    return jsonify({"message": f"Refreshed camera streams. Restarted {restarted} processes."}), 200

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    shared_namespace = manager.Namespace()
    shared_namespace.body_database_embeddings = None
    shared_namespace.body_database_names = []
    body_database_dict, embeddings_list, names_list = load_body_database(shared_namespace)
    body_database = manager.dict(body_database_dict)
    face_database = manager.dict(load_face_database())

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
    rois = manager.list([[] for _ in all_camera_urls])
    latest_frames = manager.dict()
    latest_raw_frames = manager.dict()
    latest_detections = manager.dict()
    cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
    cache_thread.start()
    initialize_queues()
    processes = []
    for i, url in enumerate(all_camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, body_database, face_database, shared_namespace),
                       daemon=True)
        processes.append(p)
        p.start()
    consumer_threads = []
    for i in range(len(all_camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames, latest_raw_frames, latest_detections),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()
    server_thread.join()