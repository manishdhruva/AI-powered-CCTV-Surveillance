
import cv2
import os
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

from flask import Flask, Response, render_template, request, redirect, url_for, session, flash
import threading

from functools import wraps

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Simulated user database (replace with a real database in production)
USERS = {
    "admin": "password123"  # Username: admin, Password: password123
}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=>> Device Used for Inference: {device}")

# Initialize YOLO model
model = YOLO("yolo11n.engine", task='detect')

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

def get_person_name():
    root = Tk()
    root.withdraw()
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    root.destroy()
    return name

def is_point_in_polygon(point, polygon):
    """Return True if point is inside polygon."""
    return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0

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

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crop, frame, shared_data, database):
    """
    Process a single detection when manually adding a new person.
    Only process the detection with the designated new-person track_id.
    """
    current_view = shared_data['current_view'].value
    view_direction = shared_data['view_directions'][current_view]
    cv2.putText(frame, f"Capture {view_direction} View", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    if shared_data['capture_signal'].value:
        shared_data['capture_images'].append(crop)
        print(f"Captured {view_direction} view.")
        shared_data['current_view'].value += 1
        shared_data['capture_signal'].value = False

        # Once all views are captured, add the person to the database.
        if shared_data['current_view'].value == len(shared_data['view_directions']):
            embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
            shared_data['adding_new_person'].value = False
            # shared_data['capture_images'].clear()
            shared_data['capture_images'][:] = []
            shared_data['current_view'].value = 0
            # Reset the track ID used for new-person capture.
            if 'new_person_track_id' in shared_data:
                del shared_data['new_person_track_id']
    return frame

def process_dynamic_new_person(crop, frame, shared_data, frame_count, database):
    """
    Process a detection when dynamically adding a new person.
    A frame counter is used to capture images every few frames.
    """
    cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    frame_count += 1

    if frame_count % 5 == 0:
        shared_data['capture_images'].append(crop)
        print(f"Dynamic capture: frame {frame_count}")

    if frame_count >= 200:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        shared_data['dynamic_adding_new_person'].value = False
        # shared_data['capture_images'].clear()
        shared_data['capture_images'][:] = []
        frame_count = 0

    return frame, frame_count

def process_standard_detections(detections, frame, roi_points, movement_trails, database):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 2)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]
            if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} entered the ROI.")
            elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} exited the ROI.")
                
    return frame, movement_trails

#########################################
# FLASK FUNCTIONS                       #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def gen_frames(camera_id, frame_queue):
    """Generator function to yield frames for streaming."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # Control frame rate

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
    """Render the main page with paginated camera feeds."""
    cameras_per_page = 2
    total_cameras = len(camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

    # Ensure page is within bounds
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    start_idx = (page - 1) * cameras_per_page
    end_idx = min(start_idx + cameras_per_page, total_cameras)
    camera_indices = list(range(start_idx, end_idx))

    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    """Stream video feed for a specific camera."""
    if camera_id >= len(camera_urls):
        return "Camera not found", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, roi_points, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Camera {camera_id} failed to open.")
        return

    # Dictionary to store movement trails for each track_id.
    movement_trails = {}
    frame_count = 0
    exter_frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"Error: Camera {camera_id} failed to read frame.")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            continue

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.4)
        # print('Frame Count: ', exter_frame_count)
        exter_frame_count+=1
        
        # Separate detections into groups.
        new_person_detections = []
        dynamic_new_person_detections = []
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            # If track IDs are not provided, default to -1.
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                # Decide which bucket this detection falls into.
                if shared_data['adding_new_person'].value and camera_id == 0:
                    # Use a dedicated track_id for new-person capture.
                    # if 'new_person_track_id' not in shared_data:
                    #     shared_data['new_person_track_id'] = track_id
                    # if track_id == shared_data['new_person_track_id']:
                    new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                elif shared_data['dynamic_adding_new_person'].value and camera_id in [2, 3]:
                    dynamic_new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                else:
                    standard_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })

        # Process the new person addition (manual capture) if active.
        if new_person_detections:
            # We assume only one candidate (the one with the new_person_track_id).
            detection = new_person_detections[0]
            frame = process_new_person(detection['crop'], frame, shared_data, database)
        # Process dynamic new person addition if active.
        elif dynamic_new_person_detections:
            detection = dynamic_new_person_detections[0]
            frame, frame_count = process_dynamic_new_person(detection['crop'], frame, shared_data, frame_count, database)
        # Process normal detections in batch.
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database)
        
        # Draw the ROI boundary.
        cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 255), thickness=2)
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"]
    })

    # Define your camera URLs.
    camera_urls = [
        # "rtsp://192.168.1.168/stream_0",
        "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
        "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
        "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
        "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
        "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
        # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
        # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
    ]

    # Draw ROI for each camera.
    rois = []
    for i, url in enumerate(camera_urls):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        success, frame = cap.read()
        cap.release()
        if not success:
            print(f"Error: Camera {i} failed to capture frame.")
            continue
        print(f"Draw ROI for Camera {i}, press 'q' when done.")
        rois.append(draw_roi(frame))


    frame_queues = [mp.Queue(maxsize=20) for _ in camera_urls]
    processes = [
        mp.Process(target=process_camera_feed,
                   args=(url, i, rois[i], frame_queues[i], shared_data, database),
                   daemon=True)
        for i, url in enumerate(camera_urls)
    ]

    for p in processes:
        p.start()


    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()


    # Run Flask app with debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

    while True:
        # for i in range(len(camera_urls)):
        #     if not frame_queues[i].empty():
        #         cv2.imshow(f"Camera {i}", frame_queues[i].get())
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # cv2.destroyAllWindows()
    for p in processes:
        p.terminate()
        p.join()











#===============================================================================================================











import cv2
import os
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

from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import threading

from functools import wraps

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Simulated user database (replace with a real database in production)
USERS = {
    "admin": "password123"  # Username: admin, Password: password123
}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=>> Device Used for Inference: {device}")

# Initialize YOLO model
model = YOLO("yolo11n.engine", task='detect')

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crop, frame, shared_data, database):
    """
    Process a single detection when manually adding a new person.
    Only process the detection with the designated new-person track_id.
    """
    current_view = shared_data['current_view'].value
    view_direction = shared_data['view_directions'][current_view]
    cv2.putText(frame, f"Capture {view_direction} View", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    if shared_data['capture_signal'].value:
        shared_data['capture_images'].append(crop)
        print(f"Captured {view_direction} view.")
        shared_data['current_view'].value += 1
        shared_data['capture_signal'].value = False

        # Once all views are captured, add the person to the database.
        if shared_data['current_view'].value == len(shared_data['view_directions']):
            embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
            shared_data['adding_new_person'].value = False
            # shared_data['capture_images'].clear()
            shared_data['capture_images'][:] = []
            shared_data['current_view'].value = 0
            # Reset the track ID used for new-person capture.
            if 'new_person_track_id' in shared_data:
                del shared_data['new_person_track_id']
    return frame

def process_dynamic_new_person(crop, frame, shared_data, frame_count, database):
    """
    Process a detection when dynamically adding a new person.
    A frame counter is used to capture images every few frames.
    """
    cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    frame_count += 1

    if frame_count % 5 == 0:
        shared_data['capture_images'].append(crop)
        print(f"Dynamic capture: frame {frame_count}")

    if frame_count >= 200:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        shared_data['dynamic_adding_new_person'].value = False
        # shared_data['capture_images'].clear()
        shared_data['capture_images'][:] = []
        frame_count = 0

    return frame, frame_count

def process_standard_detections(detections, frame, roi_points, movement_trails, database):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 2)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]
            if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} entered the ROI.")
            elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} exited the ROI.")
                
    return frame, movement_trails

#########################################
# FLASK FUNCTIONS                       #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# def gen_frames(camera_id, frame_queue):
#     """Generator function to yield frames for streaming."""
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         time.sleep(0.03)  # Control frame rate

def gen_frames(camera_id, frame_queue, view_mode):
    last_frame = None
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
            last_frame = frame
            # Resize based on view mode
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        elif last_frame is not None:
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            frame = cv2.resize(last_frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

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

# @app.route('/', defaults={'page': 1})
# @app.route('/<int:page>')
# @login_required
# def index(page):
#     """Render the main page with paginated camera feeds."""
#     cameras_per_page = 2
#     total_cameras = len(camera_urls)
#     total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

#     # Ensure page is within bounds
#     if page < 1:
#         page = 1
#     elif page > total_pages:
#         page = total_pages

#     start_idx = (page - 1) * cameras_per_page
#     end_idx = min(start_idx + cameras_per_page, total_cameras)
#     camera_indices = list(range(start_idx, end_idx))

#     return render_template('index.html', 
#                           camera_indices=camera_indices, 
#                           current_page=page, 
#                           total_pages=total_pages)

# @app.route('/', defaults={'page': 1})
# @app.route('/<int:page>')
# @login_required
# def index(page):
#     view_mode = request.args.get('view', 'separate')  # Default to 'separate'
#     cameras_per_page = 2 if view_mode == 'separate' else len(camera_urls)
#     total_cameras = len(camera_urls)
#     total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

#     if page < 1:
#         page = 1
#     elif page > total_pages:
#         page = total_pages

#     if view_mode == 'single':
#         camera_indices = list(range(total_cameras))  # Show all cameras
#     else:
#         start_idx = (page - 1) * cameras_per_page
#         end_idx = min(start_idx + cameras_per_page, total_cameras)
#         camera_indices = list(range(start_idx, end_idx))

#     return render_template('index.html', 
#                           camera_indices=camera_indices, 
#                           current_page=page, 
#                           total_pages=total_pages,
#                           view_mode=view_mode)

@app.route('/', defaults={'page': 1})
@app.route('/<int:page>')
@login_required
def index(page):
    view_mode = request.args.get('view', 'separate')
    cameras_per_page = 2 if view_mode == 'separate' else len(camera_urls)
    total_cameras = len(camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    if view_mode == 'single':
        camera_indices = list(range(total_cameras))
    else:
        start_idx = (page - 1) * cameras_per_page
        end_idx = min(start_idx + cameras_per_page, total_cameras)
        camera_indices = list(range(start_idx, end_idx))

    logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}")
    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages,
                          view_mode=view_mode)

# @app.route('/video_feed/<int:camera_id>')
# @login_required
# def video_feed(camera_id):
#     if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
#         logger.error(f"Invalid camera_id: {camera_id}")
#         return "Camera not found or not initialized", 404
#     return Response(gen_frames(camera_id, frame_queues[camera_id]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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

#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # print(f"Error: Camera {camera_id} failed to open.")
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    # Dictionary to store movement trails for each track_id.
    movement_trails = {}
    frame_count = 0
    exter_frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # print(f"Error: Camera {camera_id} failed to read frame.")
            logger.warning(f"Camera {camera_id} failed to read frame, retrying...")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            continue

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.4)
        # print('Frame Count: ', exter_frame_count)
        exter_frame_count+=1
        
        # Separate detections into groups.
        new_person_detections = []
        dynamic_new_person_detections = []
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            # If track IDs are not provided, default to -1.
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                # Decide which bucket this detection falls into.
                if shared_data['adding_new_person'].value and camera_id == 0:
                    # Use a dedicated track_id for new-person capture.
                    # if 'new_person_track_id' not in shared_data:
                    #     shared_data['new_person_track_id'] = track_id
                    # if track_id == shared_data['new_person_track_id']:
                    new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                elif shared_data['dynamic_adding_new_person'].value and camera_id in [2, 3]:
                    dynamic_new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                else:
                    standard_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })

        # Process the new person addition (manual capture) if active.
        if new_person_detections:
            # We assume only one candidate (the one with the new_person_track_id).
            detection = new_person_detections[0]
            frame = process_new_person(detection['crop'], frame, shared_data, database)
        # Process dynamic new person addition if active.
        elif dynamic_new_person_detections:
            detection = dynamic_new_person_detections[0]
            frame, frame_count = process_dynamic_new_person(detection['crop'], frame, shared_data, frame_count, database)
        
        roi_points = shared_rois[camera_id]  # Get current ROI from shared list
        # Process normal detections in batch.
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database)
        
        if roi_points and len(roi_points) >= 3:  # Ensure valid polygon
            # print(f"Drawing ROI for Camera {camera_id}: {roi_points}")
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   GLOBAL VARIABLES                    #
#########################################

# Global variables
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

def initialize_queues():
    global frame_queues
    if not frame_queues:
        print("Initializing frame queues...")
        frame_queues.extend([mp.Queue(maxsize=100) for _ in range(len(camera_urls))])
        # print(f"Initialized {len(frame_queues)} frame queues.")
        logger.info(f"Initialized {len(frame_queues)} frame queues.")

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"]
    })

    # Use a shared list for ROIs
    rois = manager.list([[] for _ in camera_urls])  # Shared across processes

    # Define your camera URLs.
    # camera_urls = [
    #     # "rtsp://192.168.1.168/stream_0",
    #     "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    #     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
    #     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
    # ]

    # Draw ROI for each camera.
    # rois = []
    # for i, url in enumerate(camera_urls):
    #     cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    #     success, frame = cap.read()
    #     cap.release()
    #     if not success:
    #         print(f"Error: Camera {i} failed to capture frame.")
    #         continue
    #     print(f"Draw ROI for Camera {i}, press 'q' when done.")
    #     rois.append(draw_roi(frame))


    # frame_queues = [mp.Queue(maxsize=20) for _ in camera_urls]

    initialize_queues()
    
    # processes = [
    #     mp.Process(target=process_camera_feed,
    #                args=(url, i, rois, frame_queues[i], shared_data, database),
    #                daemon=True)
    #     for i, url in enumerate(camera_urls)
    # ]

    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)

    # for p in processes:
        p.start()

        logger.info(f"Started process for Camera {i}: {url}")

    # Verify all processes started
    for i, p in enumerate(processes):
        if not p.is_alive():
            logger.error(f"Process for Camera {i} failed to start")


    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()


    # Run Flask app with debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

    while True:
        # for i in range(len(camera_urls)):
        #     if not frame_queues[i].empty():
        #         cv2.imshow(f"Camera {i}", frame_queues[i].get())
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # cv2.destroyAllWindows()
    if 'processes' in locals():
        for p in processes:
            p.terminate()
            p.join()












# ===============================================================================================================================

















import cv2
import os
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

from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import threading

from functools import wraps

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Simulated user database (replace with a real database in production)
USERS = {
    "admin": "password123"  # Username: admin, Password: password123
}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=>> Device Used for Inference: {device}")

# Initialize YOLO model
model = YOLO("yolo11n.engine", task='detect')

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crop, frame, shared_data, database):
    """
    Process a single detection when manually adding a new person.
    Only process the detection with the designated new-person track_id.
    """
    current_view = shared_data['current_view'].value
    view_direction = shared_data['view_directions'][current_view]
    cv2.putText(frame, f"Capture {view_direction} View", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    if shared_data['capture_signal'].value:
        shared_data['capture_images'].append(crop)
        print(f"Captured {view_direction} view.")
        shared_data['current_view'].value += 1
        shared_data['capture_signal'].value = False

        # Once all views are captured, add the person to the database.
        if shared_data['current_view'].value == len(shared_data['view_directions']):
            embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
            shared_data['adding_new_person'].value = False
            # shared_data['capture_images'].clear()
            shared_data['capture_images'][:] = []
            shared_data['current_view'].value = 0
            # Reset the track ID used for new-person capture.
            if 'new_person_track_id' in shared_data:
                del shared_data['new_person_track_id']
    return frame

def process_dynamic_new_person(crop, frame, shared_data, frame_count, database):
    """
    Process a detection when dynamically adding a new person.
    A frame counter is used to capture images every few frames.
    """
    cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    frame_count += 1

    if frame_count % 5 == 0:
        shared_data['capture_images'].append(crop)
        print(f"Dynamic capture: frame {frame_count}")

    if frame_count >= 200:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        shared_data['dynamic_adding_new_person'].value = False
        # shared_data['capture_images'].clear()
        shared_data['capture_images'][:] = []
        frame_count = 0

    return frame, frame_count

def process_standard_detections(detections, frame, roi_points, movement_trails, database):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 2)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]
            if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} entered the ROI.")
            elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} exited the ROI.")
                
    return frame, movement_trails

#########################################
# FLASK FUNCTIONS                       #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# def gen_frames(camera_id, frame_queue):
#     """Generator function to yield frames for streaming."""
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         time.sleep(0.03)  # Control frame rate

def gen_frames(camera_id, frame_queue, view_mode):
    last_frame = None
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
            last_frame = frame
            # Resize based on view mode
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        elif last_frame is not None:
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            frame = cv2.resize(last_frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

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

# @app.route('/', defaults={'page': 1})
# @app.route('/<int:page>')
# @login_required
# def index(page):
#     """Render the main page with paginated camera feeds."""
#     cameras_per_page = 2
#     total_cameras = len(camera_urls)
#     total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

#     # Ensure page is within bounds
#     if page < 1:
#         page = 1
#     elif page > total_pages:
#         page = total_pages

#     start_idx = (page - 1) * cameras_per_page
#     end_idx = min(start_idx + cameras_per_page, total_cameras)
#     camera_indices = list(range(start_idx, end_idx))

#     return render_template('index.html', 
#                           camera_indices=camera_indices, 
#                           current_page=page, 
#                           total_pages=total_pages)

# @app.route('/', defaults={'page': 1})
# @app.route('/<int:page>')
# @login_required
# def index(page):
#     view_mode = request.args.get('view', 'separate')  # Default to 'separate'
#     cameras_per_page = 2 if view_mode == 'separate' else len(camera_urls)
#     total_cameras = len(camera_urls)
#     total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

#     if page < 1:
#         page = 1
#     elif page > total_pages:
#         page = total_pages

#     if view_mode == 'single':
#         camera_indices = list(range(total_cameras))  # Show all cameras
#     else:
#         start_idx = (page - 1) * cameras_per_page
#         end_idx = min(start_idx + cameras_per_page, total_cameras)
#         camera_indices = list(range(start_idx, end_idx))

#     return render_template('index.html', 
#                           camera_indices=camera_indices, 
#                           current_page=page, 
#                           total_pages=total_pages,
#                           view_mode=view_mode)

@app.route('/', defaults={'page': 1})
@app.route('/<int:page>')
@login_required
def index(page):
    view_mode = request.args.get('view', 'separate')
    cameras_per_page = 2 if view_mode == 'separate' else len(camera_urls)
    total_cameras = len(camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    if view_mode == 'single':
        camera_indices = list(range(total_cameras))
    else:
        start_idx = (page - 1) * cameras_per_page
        end_idx = min(start_idx + cameras_per_page, total_cameras)
        camera_indices = list(range(start_idx, end_idx))

    logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}")
    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages,
                          view_mode=view_mode)

# @app.route('/video_feed/<int:camera_id>')
# @login_required
# def video_feed(camera_id):
#     if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
#         logger.error(f"Invalid camera_id: {camera_id}")
#         return "Camera not found or not initialized", 404
#     return Response(gen_frames(camera_id, frame_queues[camera_id]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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

#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # print(f"Error: Camera {camera_id} failed to open.")
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    # Dictionary to store movement trails for each track_id.
    movement_trails = {}
    frame_count = 0
    exter_frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # print(f"Error: Camera {camera_id} failed to read frame.")
            logger.warning(f"Camera {camera_id} failed to read frame, retrying...")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            continue

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.4)
        # print('Frame Count: ', exter_frame_count)
        exter_frame_count+=1
        
        # Separate detections into groups.
        new_person_detections = []
        dynamic_new_person_detections = []
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            # If track IDs are not provided, default to -1.
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                # Decide which bucket this detection falls into.
                if shared_data['adding_new_person'].value and camera_id == 0:
                    # Use a dedicated track_id for new-person capture.
                    # if 'new_person_track_id' not in shared_data:
                    #     shared_data['new_person_track_id'] = track_id
                    # if track_id == shared_data['new_person_track_id']:
                    new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                elif shared_data['dynamic_adding_new_person'].value and camera_id in [2, 3]:
                    dynamic_new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                else:
                    standard_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })

        # Process the new person addition (manual capture) if active.
        if new_person_detections:
            # We assume only one candidate (the one with the new_person_track_id).
            detection = new_person_detections[0]
            frame = process_new_person(detection['crop'], frame, shared_data, database)
        # Process dynamic new person addition if active.
        elif dynamic_new_person_detections:
            detection = dynamic_new_person_detections[0]
            frame, frame_count = process_dynamic_new_person(detection['crop'], frame, shared_data, frame_count, database)
        
        roi_points = shared_rois[camera_id]  # Get current ROI from shared list
        # Process normal detections in batch.
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database)
        
        if roi_points and len(roi_points) >= 3:  # Ensure valid polygon
            # print(f"Drawing ROI for Camera {camera_id}: {roi_points}")
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   GLOBAL VARIABLES                    #
#########################################

# Global variables
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

def initialize_queues():
    global frame_queues
    if not frame_queues:
        print("Initializing frame queues...")
        frame_queues.extend([mp.Queue(maxsize=100) for _ in range(len(camera_urls))])
        # print(f"Initialized {len(frame_queues)} frame queues.")
        logger.info(f"Initialized {len(frame_queues)} frame queues.")

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"]
    })

    # Use a shared list for ROIs
    rois = manager.list([[] for _ in camera_urls])  # Shared across processes

    # Define your camera URLs.
    # camera_urls = [
    #     # "rtsp://192.168.1.168/stream_0",
    #     "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    #     "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    #     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
    #     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
    # ]

    # Draw ROI for each camera.
    # rois = []
    # for i, url in enumerate(camera_urls):
    #     cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    #     success, frame = cap.read()
    #     cap.release()
    #     if not success:
    #         print(f"Error: Camera {i} failed to capture frame.")
    #         continue
    #     print(f"Draw ROI for Camera {i}, press 'q' when done.")
    #     rois.append(draw_roi(frame))


    # frame_queues = [mp.Queue(maxsize=20) for _ in camera_urls]

    initialize_queues()
    
    # processes = [
    #     mp.Process(target=process_camera_feed,
    #                args=(url, i, rois, frame_queues[i], shared_data, database),
    #                daemon=True)
    #     for i, url in enumerate(camera_urls)
    # ]

    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)

    # for p in processes:
        p.start()

        logger.info(f"Started process for Camera {i}: {url}")

    # Verify all processes started
    for i, p in enumerate(processes):
        if not p.is_alive():
            logger.error(f"Process for Camera {i} failed to start")


    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()


    # Run Flask app with debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

    while True:
        # for i in range(len(camera_urls)):
        #     if not frame_queues[i].empty():
        #         cv2.imshow(f"Camera {i}", frame_queues[i].get())
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # cv2.destroyAllWindows()
    if 'processes' in locals():
        for p in processes:
            p.terminate()
            p.join()

















# =======================================================================================================












import cv2
import os
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

from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import threading

from functools import wraps

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Simulated user database (replace with a real database in production)
USERS = {"admin": "password123"}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=>> Device Used for Inference: {device}")

# Initialize YOLO model
model = YOLO("yolo11n.engine", task='detect')

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crop, frame, shared_data, database):
    """
    Process a single detection when manually adding a new person.
    Only process the detection with the designated new-person track_id.
    """
    current_view = shared_data['current_view'].value
    view_direction = shared_data['view_directions'][current_view]
    cv2.putText(frame, f"Capture {view_direction} View", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    if shared_data['capture_signal'].value:
        shared_data['capture_images'].append(crop)
        print(f"Captured {view_direction} view.")
        shared_data['current_view'].value += 1
        shared_data['capture_signal'].value = False

        # Once all views are captured, add the person to the database.
        if shared_data['current_view'].value == len(shared_data['view_directions']):
            embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
            shared_data['adding_new_person'].value = False
            # shared_data['capture_images'].clear()
            shared_data['capture_images'][:] = []
            shared_data['current_view'].value = 0
            # Reset the track ID used for new-person capture.
            if 'new_person_track_id' in shared_data:
                del shared_data['new_person_track_id']
    return frame

def process_dynamic_new_person(crop, frame, shared_data, frame_count, database):
    """
    Process a detection when dynamically adding a new person.
    A frame counter is used to capture images every few frames.
    """
    cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    frame_count += 1

    if frame_count % 5 == 0:
        shared_data['capture_images'].append(crop)
        print(f"Dynamic capture: frame {frame_count}")

    if frame_count >= 200:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        shared_data['dynamic_adding_new_person'].value = False
        # shared_data['capture_images'].clear()
        shared_data['capture_images'][:] = []
        frame_count = 0

    return frame, frame_count

def process_standard_detections(detections, frame, roi_points, movement_trails, database):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 2)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]
            if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} entered the ROI.")
            elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
                print(f"ALERT: Person {label} exited the ROI.")
                
    return frame, movement_trails

#########################################
# FLASK FUNCTIONS                       #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def frame_consumer(camera_id, frame_queue, latest_frames):
    """Continuously consume frames from the queue and update latest_frames."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
        time.sleep(0.01)  # Prevent tight loop, adjust as needed

def gen_frames(camera_id, frame_queue, view_mode):
    while True:
        frame = latest_frames.get(camera_id)
        if frame is not None:
            # Resize based on view mode
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        else:
            # Display offline placeholder if no frame is available yet
            target_size = (400, 225) if view_mode == 'single' else (854, 480)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # Control streaming rate

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
    cameras_per_page = 2 if view_mode == 'separate' else len(camera_urls)
    total_cameras = len(camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    if view_mode == 'single':
        camera_indices = list(range(total_cameras))
    else:
        start_idx = (page - 1) * cameras_per_page
        end_idx = min(start_idx + cameras_per_page, total_cameras)
        camera_indices = list(range(start_idx, end_idx))

    logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}")
    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages,
                          view_mode=view_mode)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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

#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # print(f"Error: Camera {camera_id} failed to open.")
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    # Dictionary to store movement trails for each track_id.
    movement_trails = {}
    frame_count = 0
    exter_frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # print(f"Error: Camera {camera_id} failed to read frame.")
            logger.warning(f"Camera {camera_id} failed to read frame, retrying...")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            continue

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.4)
        # print('Frame Count: ', exter_frame_count)
        exter_frame_count+=1
        
        # Separate detections into groups.
        new_person_detections = []
        dynamic_new_person_detections = []
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            # If track IDs are not provided, default to -1.
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                # Decide which bucket this detection falls into.
                if shared_data['adding_new_person'].value and camera_id == 0:
                    # Use a dedicated track_id for new-person capture.
                    # if 'new_person_track_id' not in shared_data:
                    #     shared_data['new_person_track_id'] = track_id
                    # if track_id == shared_data['new_person_track_id']:
                    new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                elif shared_data['dynamic_adding_new_person'].value and camera_id in [2, 3]:
                    dynamic_new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                else:
                    standard_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })

        # Process the new person addition (manual capture) if active.
        if new_person_detections:
            # We assume only one candidate (the one with the new_person_track_id).
            detection = new_person_detections[0]
            frame = process_new_person(detection['crop'], frame, shared_data, database)
        # Process dynamic new person addition if active.
        elif dynamic_new_person_detections:
            detection = dynamic_new_person_detections[0]
            frame, frame_count = process_dynamic_new_person(detection['crop'], frame, shared_data, frame_count, database)
        
        roi_points = shared_rois[camera_id]  # Get current ROI from shared list
        # Process normal detections in batch.
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database)
        
        if roi_points and len(roi_points) >= 3:  # Ensure valid polygon
            # print(f"Drawing ROI for Camera {camera_id}: {roi_points}")
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   GLOBAL VARIABLES                    #
#########################################

# Global variables
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

def initialize_queues():
    global frame_queues
    if not frame_queues:
        print("Initializing frame queues...")
        frame_queues.extend([mp.Queue(maxsize=100) for _ in range(len(camera_urls))])
        # print(f"Initialized {len(frame_queues)} frame queues.")
        logger.info(f"Initialized {len(frame_queues)} frame queues.")

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"]
    })

    # Shared ROIs and latest frames
    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()  # Shared dictionary for latest frames

    initialize_queues()

    # Start camera processing processes
    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)
        p.start()
        logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    # Main loop for keyboard interaction
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # Cleanup
    for p in processes:
        p.terminate()
        p.join()












# ====================================================================================








import cv2
import os
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

from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import threading

from functools import wraps

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Simulated user database (replace with a real database in production)
USERS = {"admin": "password123"}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=>> Device Used for Inference: {device}")

# Initialize YOLO model
model = YOLO("yolo11n.engine", task='detect')

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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
        frame_queues.extend([mp.Queue(maxsize=100) for _ in range(len(camera_urls))])
        # print(f"Initialized {len(frame_queues)} frame queues.")
        logger.info(f"Initialized {len(frame_queues)} frame queues.")

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crop, frame, shared_data, database):
    """
    Process a single detection when manually adding a new person.
    Only process the detection with the designated new-person track_id.
    """
    current_view = shared_data['current_view'].value
    view_direction = shared_data['view_directions'][current_view]
    cv2.putText(frame, f"Capture {view_direction} View", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    if shared_data['capture_signal'].value:
        shared_data['capture_images'].append(crop)
        print(f"Captured {view_direction} view.")
        shared_data['current_view'].value += 1
        shared_data['capture_signal'].value = False

        # Once all views are captured, add the person to the database.
        if shared_data['current_view'].value == len(shared_data['view_directions']):
            embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
            shared_data['adding_new_person'].value = False
            # shared_data['capture_images'].clear()
            shared_data['capture_images'][:] = []
            shared_data['current_view'].value = 0
            # Reset the track ID used for new-person capture.
            if 'new_person_track_id' in shared_data:
                del shared_data['new_person_track_id']
    return frame

def process_dynamic_new_person(crop, frame, shared_data, frame_count, database):
    """
    Process a detection when dynamically adding a new person.
    A frame counter is used to capture images every few frames.
    """
    cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    frame_count += 1

    if frame_count % 5 == 0:
        shared_data['capture_images'].append(crop)
        print(f"Dynamic capture: frame {frame_count}")

    if frame_count >= 200:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        shared_data['dynamic_adding_new_person'].value = False
        # shared_data['capture_images'].clear()
        shared_data['capture_images'][:] = []
        frame_count = 0

    return frame, frame_count

def process_standard_detections(detections, frame, roi_points, movement_trails, database, camera_id):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 2)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]
            if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
                print(f"Camera {camera_id}, ALERT: Person {label} entered the ROI.")
            elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
                print(f"Camera {camera_id}, ALERT: Person {label} exited the ROI.")
                
    return frame, movement_trails

#########################################
# FLASK FUNCTIONS                       #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def frame_consumer(camera_id, frame_queue, latest_frames):
    """Continuously consume frames from the queue and update latest_frames."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
        time.sleep(0.01)  # Prevent tight loop, adjust as needed

def gen_frames(camera_id, frame_queue, view_mode):
    while True:
        frame = latest_frames.get(camera_id)
        if frame is not None:
            # Resize based on view mode
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        else:
            # Display offline placeholder if no frame is available yet
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # Control streaming rate

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
    
    # Define cameras per page based on view mode
    if view_mode == 'separate':
        cameras_per_page = 2  # 2 cameras per page in separate mode
    else:  # single mode
        cameras_per_page = 6  # 6 cameras per page in single mode
    
    total_cameras = len(camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

    # Ensure page is within bounds
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    # Calculate camera indices for the current page
    start_idx = (page - 1) * cameras_per_page
    end_idx = min(start_idx + cameras_per_page, total_cameras)
    camera_indices = list(range(start_idx, end_idx))

    logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}")
    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages,
                          view_mode=view_mode)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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

#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # print(f"Error: Camera {camera_id} failed to open.")
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    # Dictionary to store movement trails for each track_id.
    movement_trails = {}
    frame_count = 0
    exter_frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # print(f"Error: Camera {camera_id} failed to read frame.")
            logger.warning(f"Camera {camera_id} failed to read frame, retrying...")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            continue

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.4)
        # print('Frame Count: ', exter_frame_count)
        exter_frame_count+=1
        
        # Separate detections into groups.
        new_person_detections = []
        dynamic_new_person_detections = []
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            # If track IDs are not provided, default to -1.
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                # Decide which bucket this detection falls into.
                if shared_data['adding_new_person'].value and camera_id == 0:
                    # Use a dedicated track_id for new-person capture.
                    # if 'new_person_track_id' not in shared_data:
                    #     shared_data['new_person_track_id'] = track_id
                    # if track_id == shared_data['new_person_track_id']:
                    new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                elif shared_data['dynamic_adding_new_person'].value and camera_id in [2, 3]:
                    dynamic_new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                else:
                    standard_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })

        # Process the new person addition (manual capture) if active.
        if new_person_detections:
            # We assume only one candidate (the one with the new_person_track_id).
            detection = new_person_detections[0]
            frame = process_new_person(detection['crop'], frame, shared_data, database)
        # Process dynamic new person addition if active.
        elif dynamic_new_person_detections:
            detection = dynamic_new_person_detections[0]
            frame, frame_count = process_dynamic_new_person(detection['crop'], frame, shared_data, frame_count, database)
        
        roi_points = shared_rois[camera_id]  # Get current ROI from shared list
        # Process normal detections in batch.
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database, camera_id)
        
        if roi_points and len(roi_points) >= 3:  # Ensure valid polygon
            # print(f"Drawing ROI for Camera {camera_id}: {roi_points}")
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   GLOBAL VARIABLES                    #
#########################################

# Global variables
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"]
    })

    # Shared ROIs and latest frames
    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()  # Shared dictionary for latest frames

    initialize_queues()

    # Start camera processing processes
    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)
        p.start()
        logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    # Main loop for keyboard interaction
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # Cleanup
    for p in processes:
        p.terminate()
        p.join()








# ====================================================================================







import cv2
import os
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
from io import BytesIO
import threading
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Simulated user database (replace with a real database in production)
USERS = {"admin": "password123"}

# Check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=>> Device Used for Inference: {device}")

# Initialize YOLO model
model = YOLO("yolo11n.engine", task='detect')

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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
        frame_queues.extend([mp.Queue(maxsize=100) for _ in range(len(camera_urls))])
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
    _, buffer = cv2.imencode('.jpg', frame)
    photo_bytes = BytesIO(buffer.tobytes())
    files = {"photo": ("alert.jpg", photo_bytes, "image/jpeg")}
    payload = {
        "chat_id": chat_id,
        "caption": caption or ""
    }
    response = requests.post(url, data=payload, files=files)
    print(response.json())

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crop, frame, shared_data, database):
    """
    Process a single detection when manually adding a new person.
    Only process the detection with the designated new-person track_id.
    """
    current_view = shared_data['current_view'].value
    view_direction = shared_data['view_directions'][current_view]
    cv2.putText(frame, f"Capture {view_direction} View", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    if shared_data['capture_signal'].value:
        shared_data['capture_images'].append(crop)
        print(f"Captured {view_direction} view.")
        shared_data['current_view'].value += 1
        shared_data['capture_signal'].value = False

        # Once all views are captured, add the person to the database.
        if shared_data['current_view'].value == len(shared_data['view_directions']):
            embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
            shared_data['adding_new_person'].value = False
            # shared_data['capture_images'].clear()
            shared_data['capture_images'][:] = []
            shared_data['current_view'].value = 0
            # Reset the track ID used for new-person capture.
            if 'new_person_track_id' in shared_data:
                del shared_data['new_person_track_id']
    return frame

def process_dynamic_new_person(crop, frame, shared_data, frame_count, database):
    """
    Process a detection when dynamically adding a new person.
    A frame counter is used to capture images every few frames.
    """
    cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    frame_count += 1

    if frame_count % 5 == 0:
        shared_data['capture_images'].append(crop)
        print(f"Dynamic capture: frame {frame_count}")

    if frame_count >= 200:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        shared_data['dynamic_adding_new_person'].value = False
        # shared_data['capture_images'].clear()
        shared_data['capture_images'][:] = []
        frame_count = 0

    return frame, frame_count

def process_standard_detections(detections, frame, roi_points, movement_trails, database, camera_id):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 2)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]

            entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
            exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)

            if entered_roi or exited_roi:
                direction = "entered" if entered_roi else "exited"
                message_alert = f"📷 Camera {camera_id} ALERT: Person {label} {direction} the ROI."
                print(message_alert)
                # send_telegram_message(message=message_alert)
                send_photo_from_frame(frame, caption=message_alert)
                
    return frame, movement_trails

#########################################
# FLASK FUNCTIONS                       #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def frame_consumer(camera_id, frame_queue, latest_frames):
    """Continuously consume frames from the queue and update latest_frames."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
        time.sleep(0.01)  # Prevent tight loop, adjust as needed

def gen_frames(camera_id, frame_queue, view_mode):
    while True:
        frame = latest_frames.get(camera_id)
        if frame is not None:
            # Resize based on view mode
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        else:
            # Display offline placeholder if no frame is available yet
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # Control streaming rate

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
    
    # Define cameras per page based on view mode
    if view_mode == 'separate':
        cameras_per_page = 2  # 2 cameras per page in separate mode
    else:  # single mode
        cameras_per_page = 6  # 6 cameras per page in single mode
    
    total_cameras = len(camera_urls)
    total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page

    # Ensure page is within bounds
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    # Calculate camera indices for the current page
    start_idx = (page - 1) * cameras_per_page
    end_idx = min(start_idx + cameras_per_page, total_cameras)
    camera_indices = list(range(start_idx, end_idx))

    logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}")
    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages,
                          view_mode=view_mode)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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

#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # print(f"Error: Camera {camera_id} failed to open.")
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    # Dictionary to store movement trails for each track_id.
    movement_trails = {}
    frame_count = 0
    exter_frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # print(f"Error: Camera {camera_id} failed to read frame.")
            logger.warning(f"Camera {camera_id} failed to read frame, retrying...")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            continue

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.4)
        # print('Frame Count: ', exter_frame_count)
        exter_frame_count+=1
        
        # Separate detections into groups.
        new_person_detections = []
        dynamic_new_person_detections = []
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            # If track IDs are not provided, default to -1.
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                # Decide which bucket this detection falls into.
                if shared_data['adding_new_person'].value and camera_id == 0:
                    # Use a dedicated track_id for new-person capture.
                    # if 'new_person_track_id' not in shared_data:
                    #     shared_data['new_person_track_id'] = track_id
                    # if track_id == shared_data['new_person_track_id']:
                    new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                elif shared_data['dynamic_adding_new_person'].value and camera_id in [2, 3]:
                    dynamic_new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                else:
                    standard_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })

        # Process the new person addition (manual capture) if active.
        if new_person_detections:
            # We assume only one candidate (the one with the new_person_track_id).
            detection = new_person_detections[0]
            frame = process_new_person(detection['crop'], frame, shared_data, database)
        # Process dynamic new person addition if active.
        elif dynamic_new_person_detections:
            detection = dynamic_new_person_detections[0]
            frame, frame_count = process_dynamic_new_person(detection['crop'], frame, shared_data, frame_count, database)
        
        roi_points = shared_rois[camera_id]  # Get current ROI from shared list
        # Process normal detections in batch.
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database, camera_id)
        
        if roi_points and len(roi_points) >= 3:  # Ensure valid polygon
            # print(f"Drawing ROI for Camera {camera_id}: {roi_points}")
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   GLOBAL VARIABLES                    #
#########################################

# Global variables
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
chat_id = '-1002599945824'

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"]
    })

    # Shared ROIs and latest frames
    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()  # Shared dictionary for latest frames

    initialize_queues()

    # Start camera processing processes
    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)
        p.start()
        logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    # Main loop for keyboard interaction
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # Cleanup
    for p in processes:
        p.terminate()
        p.join()








# ====================================================================================








import cv2
import os
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
from io import BytesIO
import threading
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    model = YOLO("yolo11n.engine", task='detect')
    # logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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
        frame_queues.extend([mp.Queue(maxsize=100) for _ in range(len(camera_urls))])
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
    _, buffer = cv2.imencode('.jpg', frame)
    photo_bytes = BytesIO(buffer.tobytes())
    files = {"photo": ("alert.jpg", photo_bytes, "image/jpeg")}
    payload = {
        "chat_id": chat_id,
        "caption": caption or ""
    }
    response = requests.post(url, data=payload, files=files)
    print(response.json())

#########################################
# Processing Functions for Each Mode    #
#########################################

def process_new_person(crop, frame, shared_data, database):
    """
    Process a single detection when manually adding a new person.
    Only process the detection with the designated new-person track_id.
    """
    current_view = shared_data['current_view'].value
    view_direction = shared_data['view_directions'][current_view]
    cv2.putText(frame, f"Capture {view_direction} View", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    if shared_data['capture_signal'].value:
        shared_data['capture_images'].append(crop)
        print(f"Captured {view_direction} view.")
        shared_data['current_view'].value += 1
        shared_data['capture_signal'].value = False

        # Once all views are captured, add the person to the database.
        if shared_data['current_view'].value == len(shared_data['view_directions']):
            embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
            database[shared_data['current_name']] = embeddings
            shared_data['adding_new_person'].value = False
            # shared_data['capture_images'].clear()
            shared_data['capture_images'][:] = []
            shared_data['current_view'].value = 0
            # Reset the track ID used for new-person capture.
            if 'new_person_track_id' in shared_data:
                del shared_data['new_person_track_id']
    return frame

def process_dynamic_new_person(crop, frame, shared_data, frame_count, database):
    """
    Process a detection when dynamically adding a new person.
    A frame counter is used to capture images every few frames.
    """
    cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    frame_count += 1

    if frame_count % 5 == 0:
        shared_data['capture_images'].append(crop)
        print(f"Dynamic capture: frame {frame_count}")

    if frame_count >= 200:
        embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
        database[shared_data['current_name']] = embeddings
        shared_data['dynamic_adding_new_person'].value = False
        # shared_data['capture_images'].clear()
        shared_data['capture_images'][:] = []
        frame_count = 0

    return frame, frame_count

def process_standard_detections(detections, frame, roi_points, movement_trails, database, camera_id):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 2)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]

            entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
            exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)

            if entered_roi or exited_roi:
                direction = "entered" if entered_roi else "exited"
                message_alert = f"📷 Camera {camera_id} ALERT: Person {label} {direction} the ROI."
                # print(message_alert)
                # send_telegram_message(message=message_alert)
                send_photo_from_frame(frame, caption=message_alert)
                
    return frame, movement_trails

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

def frame_consumer(camera_id, frame_queue, latest_frames):
    """Continuously consume frames from the queue and update latest_frames."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
        time.sleep(0.01)  # Prevent tight loop, adjust as needed

def gen_frames(camera_id, frame_queue, view_mode):
    while True:
        frame = latest_frames.get(camera_id)
        if frame is not None:
            # Resize based on view mode
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        else:
            # Display offline placeholder if no frame is available yet
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # Control streaming rate

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
    
    # Add person list to the template
    persons = list(database.keys())
    logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}, persons={persons}")
    return render_template('index.html', 
                          camera_indices=camera_indices, 
                          current_page=page, 
                          total_pages=total_pages,
                          view_mode=view_mode,
                          persons=persons)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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



@app.route('/add_person', methods=['POST'])
@login_required
def add_person():
    data = request.json
    type_ = data.get('type')
    name = data.get('name')
    if not name or not type_:
        return jsonify({"message": "Name and type are required"}), 400
    
    if type_ == 'static':
        shared_data['adding_new_person'].value = True
        shared_data['current_name'] = name
    elif type_ == 'dynamic':
        shared_data['dynamic_adding_new_person'].value = True
        shared_data['current_name'] = name
    else:
        return jsonify({"message": "Invalid addition type"}), 400
    
    return jsonify({"message": f"Started {type_} addition for {name}"}), 200

@app.route('/remove_person', methods=['POST'])
@login_required
def remove_person():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({"message": "Name is required"}), 400
    
    if name in database:
        del database[name]
        # person_dir = os.path.join(database_dir, name)
        # if os.path.exists(person_dir):
        #     shutil.rmtree(person_dir)
        return jsonify({"message": f"Removed {name} from database and deleted their images"}), 200
    return jsonify({"message": f"{name} not found in database"}), 404

@app.route('/clear_database', methods=['POST'])
@login_required
def clear_database():
    database.clear()
    # if os.path.exists(database_dir):
    #     shutil.rmtree(database_dir)
    # os.makedirs(database_dir, exist_ok=True)
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

#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # print(f"Error: Camera {camera_id} failed to open.")
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    # Dictionary to store movement trails for each track_id.
    movement_trails = {}
    frame_count = 0
    exter_frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # print(f"Error: Camera {camera_id} failed to read frame.")
            logger.warning(f"Camera {camera_id} failed to read frame, retrying...")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            continue

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.4)
        # print('Frame Count: ', exter_frame_count)
        exter_frame_count+=1
        
        # Separate detections into groups.
        new_person_detections = []
        dynamic_new_person_detections = []
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            # If track IDs are not provided, default to -1.
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                # Decide which bucket this detection falls into.
                if shared_data['adding_new_person'].value and camera_id == 0:
                    # Use a dedicated track_id for new-person capture.
                    # if 'new_person_track_id' not in shared_data:
                    #     shared_data['new_person_track_id'] = track_id
                    # if track_id == shared_data['new_person_track_id']:
                    new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                elif shared_data['dynamic_adding_new_person'].value and camera_id in [2, 3]:
                    dynamic_new_person_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })
                else:
                    standard_detections.append({
                        'crop': person_crop,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'track_id': track_id
                    })

        # Process the new person addition (manual capture) if active.
        if new_person_detections:
            # We assume only one candidate (the one with the new_person_track_id).
            detection = new_person_detections[0]
            frame = process_new_person(detection['crop'], frame, shared_data, database)
        # Process dynamic new person addition if active.
        elif dynamic_new_person_detections:
            detection = dynamic_new_person_detections[0]
            frame, frame_count = process_dynamic_new_person(detection['crop'], frame, shared_data, frame_count, database)
        
        roi_points = shared_rois[camera_id]  # Get current ROI from shared list
        # Process normal detections in batch.
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database, camera_id)
        
        if roi_points and len(roi_points) >= 3:  # Ensure valid polygon
            # print(f"Drawing ROI for Camera {camera_id}: {roi_points}")
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   GLOBAL VARIABLES                    #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

# URLs of all IP cameras
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

# Credentials for Telegram Channel/Group
bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
chat_id = '-1002599945824'

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    # logger.info(f"Initial database contents: {dict(database).keys()}")
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"]
    })

    # Shared ROIs and latest frames
    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()  # Shared dictionary for latest frames

    initialize_queues()

    # Start camera processing processes
    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)
        p.start()
        # logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        # logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    # Main loop for keyboard interaction
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # Cleanup
    for p in processes:
        p.terminate()
        p.join()








# ====================================================================================











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
from io import BytesIO
import threading
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    model = YOLO("yolo11n.engine", task='detect')
    # logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

database_dir = './database'

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
                         std=[0.229, 0.224, 0.225])
])

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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
        frame_queues.extend([mp.Queue(maxsize=1000) for _ in range(len(camera_urls))])
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

def process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold=28):
    """
    Process dynamic capture from multiple cameras, collecting crops only from detected bounding boxes.
    Stops when the total number of captured crops reaches the threshold (default 28).
    """
    frame_count = 0
    max_frames = 200  # Fallback limit to prevent infinite loop
    while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
        crops = []
        for camera_id in cameras[:4]:  # Use the 4 selected cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]
                    crops.append(person_crop)
                # No else: only append crops when person is detected
            # No else: skip if frame is None
        
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

def process_standard_detections(detections, frame, roi_points, movement_trails, database, camera_id):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 1)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]

            entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
            exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)

            if entered_roi or exited_roi:
                current_time = time.time()
                if track_id not in last_alert_time or (current_time - last_alert_time[track_id]) > 10:  # 10-second cooldown
                    direction = "entered" if entered_roi else "exited"
                    message_alert = f"📷 Camera {camera_id} ALERT: Person {label} {direction} the ROI."
                    telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
                    last_alert_time[track_id] = current_time
                
    return frame, movement_trails


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

def frame_consumer(camera_id, frame_queue, latest_frames):
    """Continuously consume frames from the queue and update latest_frames."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
            logger.debug(f"Updated latest frame for Camera {camera_id}")
        time.sleep(0.01)  # Prevent tight loop, adjust as needed

def gen_frames(camera_id, frame_queue, view_mode):
    while True:
        frame = latest_frames.get(camera_id)
        if frame is not None:
            # Resize based on view mode
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        else:
            # Display offline placeholder if no frame is available yet
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # Control streaming rate



# def gen_frames(camera_id, frame_queue, view_mode):
#     offline_timeout = 0
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             frame = cv2.resize(frame, target_size)
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             else:
#                 logger.error(f"Failed to encode frame for Camera {camera_id}")
#             offline_timeout = 0
#         else:
#             offline_timeout += 1
#             if offline_timeout > 100:  # Show offline message after ~0.5s of no frames
#                 target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#                 blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
#                 cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             time.sleep(0.005)



# def gen_frames(camera_id, frame_queue, view_mode):
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             frame = cv2.resize(frame, target_size)
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Compress to 85% quality
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             else:
#                 logger.error(f"Failed to encode frame for Camera {camera_id}")
#         else:
#             time.sleep(0.005)  # Reduced sleep for faster recovery



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
                          active_cameras=active_cameras)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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



# @app.route('/add_person', methods=['POST'])
# @login_required
# def add_person():
#     data = request.json
#     type_ = data.get('type')
#     name = data.get('name')
#     cameras = data.get('cameras', [])
#     if not name or not type_:
#         return jsonify({"message": "Name and type are required"}), 400
    
#     if type_ == 'static':
#         shared_data['adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
        
#         # Fetch and crop the latest frames
#         crops = []
#         for camera_id in cameras[:4]:  # Limit to 4 cameras
#             frame = latest_frames.get(camera_id)
#             if frame is not None:
#                 # Detect and crop using YOLOv11
#                 results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
#                 if results and results[0].boxes is not None and len(results[0].boxes) > 0:
#                     # Select the most confident detection
#                     best_box = max(results[0].boxes, key=lambda b: b.conf)
#                     # Flatten the tensor and unpack
#                     x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
#                     person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box
#                     crops.append(person_crop)
#                 else:
#                     logger.warning(f"No person detected in camera {camera_id}")
#                     shared_data['adding_new_person'].value = False
#                     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
#             else:
#                 logger.warning(f"No frame available for camera {camera_id}")
#                 shared_data['adding_new_person'].value = False
#                 return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
        
#         if len(crops) == 4:  # Ensure we have all 4 views
#             success = process_new_person(crops, shared_data, database)
#             if success:
#                 return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
#             else:
#                 return jsonify({"message": "Failed to capture all views"}), 500
#         else:
#             shared_data['adding_new_person'].value = False
#             return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
#     elif type_ == 'dynamic':
#         shared_data['dynamic_adding_new_person'].value = True
#         shared_data['current_name'] = name
#         return jsonify({"message": f"Started dynamic addition for {name}"}), 200
    
#     return jsonify({"message": "Invalid addition type"}), 400


#temporary
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
        
        # Fetch and crop the latest frames
        crops = []
        for camera_id in cameras[:4]:  # Limit to 4 cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                # Detect and crop using YOLOv11 (disabled strict check for testing)
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                person_crop = frame  # Use full frame as fallback
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box if detected
                # else:  # Temporarily disabled for testing
                #     logger.warning(f"No person detected in camera {camera_id}")
                #     shared_data['adding_new_person'].value = False
                #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
                crops.append(person_crop)
            else:
                logger.warning(f"No frame available for camera {camera_id}")
                crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))  # Blank frame for testing
        
        # Proceed even if fewer than 4 crops (for testing)
        # if len(crops) == 4:  # Temporarily disabled
        success = process_new_person(crops, shared_data, database)
        if success:
            return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
        else:
            return jsonify({"message": "Failed to capture all views"}), 500
        # else:
        #     shared_data['adding_new_person'].value = False
        #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
    elif type_ == 'dynamic':
        shared_data['dynamic_adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        
        def run_dynamic_capture():
            success = process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold)
            logger.info(f"Dynamic capture thread completed for {name}, success: {success}")
        
        threading.Thread(target=run_dynamic_capture, daemon=True).start()
        logger.info(f"Started dynamic capture thread for {name}")
        # return jsonify({"message": f"Started dynamic addition for {name}. Please wait until completion."}), 200
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
        # person_dir = os.path.join(database_dir, name)
        # if os.path.exists(person_dir):
        #     shutil.rmtree(person_dir)
        return jsonify({"message": f"Removed {name} from database and deleted their images"}), 200
    return jsonify({"message": f"{name} not found in database"}), 404

@app.route('/clear_database', methods=['POST'])
@login_required
def clear_database():
    database.clear()
    # if os.path.exists(database_dir):
    #     shutil.rmtree(database_dir)
    # os.makedirs(database_dir, exist_ok=True)
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
    
    return render_template('capture_static.html', 
                          name=name, 
                          camera_indices=camera_indices)

@app.route('/capture_dynamic', methods=['GET'])
@login_required
def capture_dynamic():
    name = request.args.get('name')
    cameras = request.args.get('cameras')  # Comma-separated list of camera indices
    if not name or not cameras:
        flash("Name and cameras are required for dynamic capture.")
        return redirect(url_for('index'))
    
    try:
        camera_indices = [int(i) for i in cameras.split(',')][:4]  # Limit to 4 cameras
        if not all(0 <= i < len(camera_urls) for i in camera_indices):
            raise ValueError("Invalid camera indices")
    except ValueError:
        flash("Invalid camera selection.")
        return redirect(url_for('index'))
    
    return render_template('capture_dynamic.html', 
                          name=name, 
                          camera_indices=camera_indices)

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


#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)  # Larger buffer for smoother streaming

    if not cap.isOpened():
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    movement_trails = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning(f"Camera {camera_id} failed to read frame, retrying in 1s...")
            time.sleep(2)  # Wait before retrying
            cap.release()
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 20)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)
            continue

        # Get detection/tracking results from YOLO
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
        
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue

            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box
                track_id = int(track_id) if track_id is not None else -1

                standard_detections.append({
                    'crop': person_crop,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'track_id': track_id
                })

        # Process standard detections
        roi_points = shared_rois[camera_id]
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database, camera_id)
        
        if roi_points and len(roi_points) >= 3:
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   GLOBAL VARIABLES                    #
#########################################

# Global dictionary to store latest frames for each camera
latest_frames = {}

# URLs of all IP cameras
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

# Credentials for Telegram Channel/Group
bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
chat_id = '-1002599945824'

# Global variable to cache active cameras
active_cameras_cache = list(range(len(camera_urls)))  # Default to all cameras initially
cache_lock = threading.Lock()

# Track last alert time per track_id
last_alert_time = {}  

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    # logger.info(f"Initial database contents: {dict(database).keys()}")

    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"],
        'selected_cameras': manager.list()
    })

    # Shared ROIs and latest frames
    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()  # Shared dictionary for latest frames

    # Start background thread to update cache
    cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
    cache_thread.start()

    initialize_queues()

    # Start camera processing processes
    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)
        p.start()
        # logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        # logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    # Main loop for keyboard interaction
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # Cleanup
    for p in processes:
        p.terminate()
        p.join()













# ====================================================================================










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
from io import BytesIO
import threading
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    model = YOLO("yolo11n.engine", task='detect')
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
                         std=[0.229, 0.224, 0.225])
])

#########################################
#   GLOBAL VARIABLES                    #
#########################################

database_dir = './database'

# Global dictionary to store latest frames for each camera
latest_frames = {}

# URLs of all IP cameras
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

# Credentials for Telegram Channel/Group
bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
chat_id = '-1002599945824'

# Global variable to cache active cameras
active_cameras_cache = list(range(len(camera_urls)))  # Default to all cameras initially
cache_lock = threading.Lock()

# Track last alert time per track_id
last_alert_time = {}  

# Global or shared variable (e.g., in main or manager.dict)
database_embeddings = None  # Cached tensor of all embeddings
database_names = []  # Corresponding names

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
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def match_embedding(query_embedding, database, threshold=0.7):
    best_match, best_similarity = None, 0
    for name, embeddings in database.items():
        for embedding in embeddings:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_match, best_similarity = name, similarity
    return best_match, best_similarity

def load_database():
    database = {}
    if os.path.exists(database_dir):
        for person_name in os.listdir(database_dir):
            person_dir = os.path.join(database_dir, person_name)
            if os.path.isdir(person_dir):
                embeddings = []
                for image_filename in os.listdir(person_dir):
                    image = cv2.imread(os.path.join(person_dir, image_filename))
                    if image is not None:
                        embeddings.append(extract_embedding(image))
                database[person_name] = embeddings
    return database

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
        frame_queues.extend([mp.Queue(maxsize=1000) for _ in range(len(camera_urls))])
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

def process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold=28):
    """
    Process dynamic capture from multiple cameras, collecting crops only from detected bounding boxes.
    Stops when the total number of captured crops reaches the threshold (default 28).
    """
    frame_count = 0
    max_frames = 200  # Fallback limit to prevent infinite loop
    while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
        crops = []
        for camera_id in cameras[:4]:  # Use the 4 selected cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]
                    crops.append(person_crop)
                # No else: only append crops when person is detected
            # No else: skip if frame is None
        
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

def process_standard_detections(detections, frame, roi_points, movement_trails, database, camera_id):
    """
    Batch process all standard detections (i.e. not adding a new person).
    Compute ReID embeddings in batch and then update the display and movement trails.
    """
    if not detections:
        return frame, movement_trails

    # Create a list of crops from the detections.
    crops = [det['crop'] for det in detections]
    # Preprocess and stack images to a batch tensor.
    transformed_images = [transform(crop) for crop in crops]
    batch_tensor = torch.stack(transformed_images).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    embeddings = embeddings.cpu().numpy()
    
    # Loop over each detection and update drawing and movement trails.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = det['center']
        track_id = det['track_id']
        query_embedding = embeddings[i]
        match, similarity = match_embedding(query_embedding, database)
        label = match if match else "Unknown"
        
        if is_point_in_polygon(center, roi_points):
            flabel = f"{label} Inside ROI ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} Outside ROI ({similarity:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, flabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Update movement trails for this track_id.
        if track_id not in movement_trails:
            movement_trails[track_id] = []
        movement_trails[track_id].append(center)
        if len(movement_trails[track_id]) > 30:
            movement_trails[track_id].pop(0)
        
        for j in range(1, len(movement_trails[track_id])):
            cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 1)
        
        # Check for ROI boundary crossing.
        if len(movement_trails[track_id]) > 1:
            last_point = movement_trails[track_id][-2]
            current_point = movement_trails[track_id][-1]

            entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
            exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)

            if entered_roi or exited_roi:
                current_time = time.time()
                if track_id not in last_alert_time or (current_time - last_alert_time[track_id]) > 10:  # 10-second cooldown
                    direction = "entered" if entered_roi else "exited"
                    message_alert = f"📷 Camera {camera_id} ALERT: Person {label} {direction} the ROI."
                    telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
                    last_alert_time[track_id] = current_time
                
    return frame, movement_trails

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

def frame_consumer(camera_id, frame_queue, latest_frames):
    """Continuously consume frames from the queue and update latest_frames."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            latest_frames[camera_id] = frame
            logger.debug(f"Updated latest frame for Camera {camera_id}")
        time.sleep(0.01)  # Prevent tight loop, adjust as needed

def gen_frames(camera_id, frame_queue, view_mode):
    while True:
        frame = latest_frames.get(camera_id)
        if frame is not None:
            # Resize based on view mode
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            frame = cv2.resize(frame, target_size)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                logger.error(f"Failed to encode frame for Camera {camera_id}")
        else:
            # Display offline placeholder if no frame is available yet
            target_size = (600, 338) if view_mode == 'single' else (1000, 563)
            blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # Control streaming rate



# def gen_frames(camera_id, frame_queue, view_mode):
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             frame = cv2.resize(frame, target_size)
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Compress to 85% quality
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             else:
#                 logger.error(f"Failed to encode frame for Camera {camera_id}")
#         else:
#             time.sleep(0.005)  # Reduced sleep for faster recovery



# def gen_frames(camera_id, frame_queue, view_mode):
#     offline_timeout = 0
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             frame = cv2.resize(frame, target_size)
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             else:
#                 logger.error(f"Failed to encode frame for Camera {camera_id}")
#             offline_timeout = 0
#         else:
#             offline_timeout += 1
#             if offline_timeout > 100:  # Show offline message after ~0.5s of no frames
#                 target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#                 blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
#                 cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             time.sleep(0.005)




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
                          active_cameras=active_cameras)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
    if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
        logger.error(f"Invalid camera_id: {camera_id}")
        return "Camera not found or not initialized", 404
    return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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



# @app.route('/add_person', methods=['POST'])
# @login_required
# def add_person():
#     data = request.json
#     type_ = data.get('type')
#     name = data.get('name')
#     cameras = data.get('cameras', [])
#     if not name or not type_:
#         return jsonify({"message": "Name and type are required"}), 400
    
#     if type_ == 'static':
#         shared_data['adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
        
#         # Fetch and crop the latest frames
#         crops = []
#         for camera_id in cameras[:4]:  # Limit to 4 cameras
#             frame = latest_frames.get(camera_id)
#             if frame is not None:
#                 # Detect and crop using YOLOv11
#                 results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
#                 if results and results[0].boxes is not None and len(results[0].boxes) > 0:
#                     # Select the most confident detection
#                     best_box = max(results[0].boxes, key=lambda b: b.conf)
#                     # Flatten the tensor and unpack
#                     x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
#                     person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box
#                     crops.append(person_crop)
#                 else:
#                     logger.warning(f"No person detected in camera {camera_id}")
#                     shared_data['adding_new_person'].value = False
#                     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
#             else:
#                 logger.warning(f"No frame available for camera {camera_id}")
#                 shared_data['adding_new_person'].value = False
#                 return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
        
#         if len(crops) == 4:  # Ensure we have all 4 views
#             success = process_new_person(crops, shared_data, database)
#             if success:
#                 return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
#             else:
#                 return jsonify({"message": "Failed to capture all views"}), 500
#         else:
#             shared_data['adding_new_person'].value = False
#             return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
#     elif type_ == 'dynamic':
#         shared_data['dynamic_adding_new_person'].value = True
#         shared_data['current_name'] = name
#         return jsonify({"message": f"Started dynamic addition for {name}"}), 200
    
#     return jsonify({"message": "Invalid addition type"}), 400


#temporary
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
        
        # Fetch and crop the latest frames
        crops = []
        for camera_id in cameras[:4]:  # Limit to 4 cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                # Detect and crop using YOLOv11 (disabled strict check for testing)
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                person_crop = frame  # Use full frame as fallback
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box if detected
                # else:  # Temporarily disabled for testing
                #     logger.warning(f"No person detected in camera {camera_id}")
                #     shared_data['adding_new_person'].value = False
                #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
                crops.append(person_crop)
            else:
                logger.warning(f"No frame available for camera {camera_id}")
                crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))  # Blank frame for testing
        
        # Proceed even if fewer than 4 crops (for testing)
        # if len(crops) == 4:  # Temporarily disabled
        success = process_new_person(crops, shared_data, database)
        if success:
            return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
        else:
            return jsonify({"message": "Failed to capture all views"}), 500
        # else:
        #     shared_data['adding_new_person'].value = False
        #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
    elif type_ == 'dynamic':
        shared_data['dynamic_adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        
        def run_dynamic_capture():
            success = process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold)
            logger.info(f"Dynamic capture thread completed for {name}, success: {success}")
        
        threading.Thread(target=run_dynamic_capture, daemon=True).start()
        logger.info(f"Started dynamic capture thread for {name}")
        # return jsonify({"message": f"Started dynamic addition for {name}. Please wait until completion."}), 200
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
        # person_dir = os.path.join(database_dir, name)
        # if os.path.exists(person_dir):
        #     shutil.rmtree(person_dir)
        return jsonify({"message": f"Removed {name} from database and deleted their images"}), 200
    return jsonify({"message": f"{name} not found in database"}), 404

@app.route('/clear_database', methods=['POST'])
@login_required
def clear_database():
    database.clear()
    # if os.path.exists(database_dir):
    #     shutil.rmtree(database_dir)
    # os.makedirs(database_dir, exist_ok=True)
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
    
    return render_template('capture_static.html', 
                          name=name, 
                          camera_indices=camera_indices)

@app.route('/capture_dynamic', methods=['GET'])
@login_required
def capture_dynamic():
    name = request.args.get('name')
    cameras = request.args.get('cameras')  # Comma-separated list of camera indices
    if not name or not cameras:
        flash("Name and cameras are required for dynamic capture.")
        return redirect(url_for('index'))
    
    try:
        camera_indices = [int(i) for i in cameras.split(',')][:4]  # Limit to 4 cameras
        if not all(0 <= i < len(camera_urls) for i in camera_indices):
            raise ValueError("Invalid camera indices")
    except ValueError:
        flash("Invalid camera selection.")
        return redirect(url_for('index'))
    
    return render_template('capture_dynamic.html', 
                          name=name, 
                          camera_indices=camera_indices)

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


#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)  # Larger buffer for smoother streaming

    if not cap.isOpened():
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    movement_trails = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning(f"Camera {camera_id} failed to read frame, retrying in 1s...")
            time.sleep(2)  # Wait before retrying
            cap.release()
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 20)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)
            continue

        # Get detection/tracking results from YOLO
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
        
        standard_detections = []
        
        for result in results:
            if result.boxes is None:
                continue

            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box
                track_id = int(track_id) if track_id is not None else -1

                standard_detections.append({
                    'crop': person_crop,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'track_id': track_id
                })

        # Process standard detections
        roi_points = shared_rois[camera_id]
        frame, movement_trails = process_standard_detections(standard_detections, frame, roi_points, movement_trails, database, camera_id)
        
        if roi_points and len(roi_points) >= 3:
            cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        
        frame_queue.put(cv2.resize(frame, (854, 480)))

    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    database = manager.dict(load_database())
    # logger.info(f"Initial database contents: {dict(database).keys()}")

    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"],
        'selected_cameras': manager.list()
    })

    # Shared ROIs and latest frames
    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()  # Shared dictionary for latest frames

    # Start background thread to update cache
    cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
    cache_thread.start()

    initialize_queues()

    # Start camera processing processes
    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database),
                       daemon=True)
        processes.append(p)
        p.start()
        # logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        # logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    # Main loop for keyboard interaction
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            database.clear()
            print("The database library has been cleaned up.")
        elif key == ord('n') and not shared_data['adding_new_person'].value:
            shared_data['adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['adding_new_person'].value = False
        elif key == ord('c'):
            shared_data['capture_signal'].value = True
        elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
            shared_data['dynamic_adding_new_person'].value = True
            shared_data['current_name'] = get_person_name()
            if not shared_data['current_name']:
                shared_data['dynamic_adding_new_person'].value = False

    # Cleanup
    for p in processes:
        p.terminate()
        p.join()










# =====================================================================================================================================













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
from io import BytesIO
import threading
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Empty  # Explicitly import the correct Empty exception

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
logger = logging.getLogger(__name__)

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
    model = YOLO("yolo11n.engine", task='detect')
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
                         std=[0.229, 0.224, 0.225])
])

#########################################
#   GLOBAL VARIABLES                    #
#########################################

database_dir = './database'

# Global dictionary to store latest frames for each camera
latest_frames = {}

# URLs of all IP cameras
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", # 
    # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
    # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
]

# rois = [[] for _ in camera_urls]  # Initialize empty ROIs for each camera
frame_queues = []

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
                logger.info(f"Loaded {image_count} images for {person_name}")
        if embeddings_list:
            # Keep tensor on CPU to avoid CUDA serialization issues
            shared_namespace.database_embeddings = torch.tensor(np.stack(embeddings_list))  # No .to(device) here
            shared_namespace.database_names = names_list
            logger.info(f"Database loaded: {len(shared_namespace.database_names)} embeddings, {len(database)} unique persons")
        else:
            logger.warning("No embeddings loaded into database_embeddings")
    else:
        logger.error(f"Database directory {database_dir} does not exist")
    return database

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

def frame_consumer(camera_id, frame_queue, latest_frames):
    while True:
        try:
            frame = frame_queue.get(timeout=1)  # Get frame with timeout
            latest_frames[camera_id] = frame
            # logger.debug(f"Updated latest frame for Camera {camera_id}")
        except Empty:  # Catch the correct Empty exception
            # logger.debug(f"Queue empty for Camera {camera_id}")
            if camera_id not in latest_frames:
                latest_frames[camera_id] = generate_offline_frame(camera_id)
            time.sleep(0.1)  # Small delay to avoid busy-waiting



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

def process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold=28):
    """
    Process dynamic capture from multiple cameras, collecting crops only from detected bounding boxes.
    Stops when the total number of captured crops reaches the threshold (default 28).
    """
    frame_count = 0
    max_frames = 200  # Fallback limit to prevent infinite loop
    while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
        crops = []
        for camera_id in cameras[:4]:  # Use the 4 selected cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]
                    crops.append(person_crop)
                # No else: only append crops when person is detected
            # No else: skip if frame is None
        
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
            flabel = f"{label} ({similarity:.2f})"
            color = (0, 255, 0)
        else:
            flabel = f"{label} ({similarity:.2f})"
            color = (0, 0, 255)

        cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, str(track_id), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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

#########################################
# FLASK FUNCTIONS                       #
#########################################


# def frame_consumer(camera_id, frame_queue, latest_frames):
#     """Continuously consume frames from the queue and update latest_frames."""
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             # logger.debug(f"Updated latest frame for Camera {camera_id}")
#         time.sleep(0.01)  # Prevent tight loop, adjust as needed

# def gen_frames(camera_id, frame_queue, view_mode):
#     while True:
#         frame = latest_frames.get(camera_id)
#         if frame is not None:
#             # Resize based on view mode
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             frame = cv2.resize(frame, target_size)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             else:
#                 logger.error(f"Failed to encode frame for Camera {camera_id}")
#         else:
#             # Display offline placeholder if no frame is available yet
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
#             cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             ret, buffer = cv2.imencode('.jpg', blank_frame)
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         time.sleep(0.01)  # Control streaming rate

# def gen_frames(camera_id, frame_queue, view_mode):
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             frame = cv2.resize(frame, target_size)
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Compress to 85% quality
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             else:
#                 logger.error(f"Failed to encode frame for Camera {camera_id}")
#         else:
#             time.sleep(0.005)  # Reduced sleep for faster recovery

# def gen_frames(camera_id, frame_queue, view_mode):
#     offline_timeout = 0
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             latest_frames[camera_id] = frame
#             target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#             frame = cv2.resize(frame, target_size)
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if ret:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             else:
#                 logger.error(f"Failed to encode frame for Camera {camera_id}")
#             offline_timeout = 0
#         else:
#             offline_timeout += 1
#             if offline_timeout > 100:  # Show offline message after ~0.5s of no frames
#                 target_size = (600, 338) if view_mode == 'single' else (1000, 563)
#                 blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
#                 cv2.putText(blank_frame, f"Camera {camera_id} Offline", (50, target_size[1]//2), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             time.sleep(0.005)


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
                          active_cameras=active_cameras)

# @app.route('/video_feed/<int:camera_id>')
# @login_required
# def video_feed(camera_id):
#     view_mode = request.args.get('view', 'separate')  # Pass view_mode to gen_frames
#     if not frame_queues or camera_id >= len(frame_queues) or camera_id < 0:
#         logger.error(f"Invalid camera_id: {camera_id}")
#         return "Camera not found or not initialized", 404
#     return Response(gen_frames(camera_id, frame_queues[camera_id], view_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

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



# @app.route('/add_person', methods=['POST'])
# @login_required
# def add_person():
#     data = request.json
#     type_ = data.get('type')
#     name = data.get('name')
#     cameras = data.get('cameras', [])
#     if not name or not type_:
#         return jsonify({"message": "Name and type are required"}), 400
    
#     if type_ == 'static':
#         shared_data['adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
        
#         # Fetch and crop the latest frames
#         crops = []
#         for camera_id in cameras[:4]:  # Limit to 4 cameras
#             frame = latest_frames.get(camera_id)
#             if frame is not None:
#                 # Detect and crop using YOLOv11
#                 results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
#                 if results and results[0].boxes is not None and len(results[0].boxes) > 0:
#                     # Select the most confident detection
#                     best_box = max(results[0].boxes, key=lambda b: b.conf)
#                     # Flatten the tensor and unpack
#                     x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
#                     person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box
#                     crops.append(person_crop)
#                 else:
#                     logger.warning(f"No person detected in camera {camera_id}")
#                     shared_data['adding_new_person'].value = False
#                     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
#             else:
#                 logger.warning(f"No frame available for camera {camera_id}")
#                 shared_data['adding_new_person'].value = False
#                 return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
        
#         if len(crops) == 4:  # Ensure we have all 4 views
#             success = process_new_person(crops, shared_data, database)
#             if success:
#                 return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
#             else:
#                 return jsonify({"message": "Failed to capture all views"}), 500
#         else:
#             shared_data['adding_new_person'].value = False
#             return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
#     elif type_ == 'dynamic':
#         shared_data['dynamic_adding_new_person'].value = True
#         shared_data['current_name'] = name
#         return jsonify({"message": f"Started dynamic addition for {name}"}), 200
    
#     return jsonify({"message": "Invalid addition type"}), 400


#temporary
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
        
        # Fetch and crop the latest frames
        crops = []
        for camera_id in cameras[:4]:  # Limit to 4 cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                # Detect and crop using YOLOv11 (disabled strict check for testing)
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                person_crop = frame  # Use full frame as fallback
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box if detected
                # else:  # Temporarily disabled for testing
                #     logger.warning(f"No person detected in camera {camera_id}")
                #     shared_data['adding_new_person'].value = False
                #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
                crops.append(person_crop)
            else:
                logger.warning(f"No frame available for camera {camera_id}")
                crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))  # Blank frame for testing
        
        # Proceed even if fewer than 4 crops (for testing)
        # if len(crops) == 4:  # Temporarily disabled
        success = process_new_person(crops, shared_data, database)
        if success:
            return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
        else:
            return jsonify({"message": "Failed to capture all views"}), 500
        # else:
        #     shared_data['adding_new_person'].value = False
        #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
    elif type_ == 'dynamic':
        shared_data['dynamic_adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        
        def run_dynamic_capture():
            success = process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold)
            logger.info(f"Dynamic capture thread completed for {name}, success: {success}")
        
        threading.Thread(target=run_dynamic_capture, daemon=True).start()
        logger.info(f"Started dynamic capture thread for {name}")
        # return jsonify({"message": f"Started dynamic addition for {name}. Please wait until completion."}), 200
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
        # person_dir = os.path.join(database_dir, name)
        # if os.path.exists(person_dir):
        #     shutil.rmtree(person_dir)
        return jsonify({"message": f"Removed {name} from database and deleted their images"}), 200
    return jsonify({"message": f"{name} not found in database"}), 404

@app.route('/clear_database', methods=['POST'])
@login_required
def clear_database():
    database.clear()
    # if os.path.exists(database_dir):
    #     shutil.rmtree(database_dir)
    # os.makedirs(database_dir, exist_ok=True)
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
    
    return render_template('capture_static.html', 
                          name=name, 
                          camera_indices=camera_indices)

@app.route('/capture_dynamic', methods=['GET'])
@login_required
def capture_dynamic():
    name = request.args.get('name')
    cameras = request.args.get('cameras')  # Comma-separated list of camera indices
    if not name or not cameras:
        flash("Name and cameras are required for dynamic capture.")
        return redirect(url_for('index'))
    
    try:
        camera_indices = [int(i) for i in cameras.split(',')][:4]  # Limit to 4 cameras
        if not all(0 <= i < len(camera_urls) for i in camera_indices):
            raise ValueError("Invalid camera indices")
    except ValueError:
        flash("Invalid camera selection.")
        return redirect(url_for('index'))
    
    return render_template('capture_dynamic.html', 
                          name=name, 
                          camera_indices=camera_indices)

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


#########################################
# MODIFIED PROCESSING FUNCTION          #
#########################################

# def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database, shared_namespace):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)

#     if not cap.isOpened():
#         logger.error(f"Camera {camera_id} failed to open: {camera_url}")
#         return
    
#     # logger.info(f"Camera {camera_id} initialized: {camera_url}")

#     movement_trails = {}
#     tracked_embeddings = {}

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             logger.warning(f"Camera {camera_id} failed to read frame, retrying in 1s...")
#             time.sleep(1)
#             cap.release()
#             cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#             cap.set(cv2.CAP_PROP_FPS, 20)
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)
#             continue

#         results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.7)
#         current_detections = []

#         for result in results:
#             if result.boxes is None:
#                 continue
#             track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
#             for box, track_id in zip(result.boxes, track_ids):
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center = ((x1 + x2) // 2, y2)
#                 person_crop = frame[y1:y2, x1:x2]
#                 track_id = int(track_id) if track_id is not None else -1

#                 embedding = extract_embedding(person_crop)
#                 tracked_embeddings[track_id] = embedding
#                 # logger.debug(f"Live embedding for track {track_id}: norm={np.linalg.norm(embedding):.2f}")

#                 current_detections.append({
#                     'crop': person_crop,
#                     'bbox': (x1, y1, x2, y2),
#                     'center': center,
#                     'track_id': track_id,
#                     'embedding': embedding
#                 })

#         roi_points = shared_rois[camera_id]
#         processed_frame = frame
#         if current_detections:
#             processed_frame, movement_trails = process_standard_detections(
#                 current_detections, processed_frame, roi_points, movement_trails, database, camera_id, shared_namespace
#             )

#         if roi_points and len(roi_points) >= 3:
#             cv2.polylines(processed_frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

#         if frame_queue.qsize() > 450:
#             frame_queue.get()
#         frame_queue.put(cv2.resize(processed_frame, (854, 480)))

#         if len(tracked_embeddings) > 100:
#             oldest_track = min(tracked_embeddings.keys(), key=lambda k: tracked_embeddings[k].size)
#             tracked_embeddings.pop(oldest_track)

#     cap.release()
#     logger.info(f"Camera {camera_id} process ended")

def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database, shared_namespace):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)

    if not cap.isOpened():
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        # Push offline frame immediately if camera can't open
        offline_frame = generate_offline_frame(camera_id)
        frame_queue.put(offline_frame)
        return
    
    # logger.info(f"Camera {camera_id} initialized: {camera_url}")

    movement_trails = {}
    tracked_embeddings = {}
    reconnect_attempts = 0
    max_attempts = 5  # Limit reconnection attempts

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning(f"Camera {camera_id} failed to read frame, attempt {reconnect_attempts + 1}/{max_attempts}")
            reconnect_attempts += 1
            if reconnect_attempts >= max_attempts:
                logger.error(f"Camera {camera_id} offline after {max_attempts} attempts")
                offline_frame = generate_offline_frame(camera_id)
                if frame_queue.qsize() > 450:
                    frame_queue.get()
                frame_queue.put(offline_frame)
                break  # Exit loop if max attempts reached

            # Attempt to reconnect
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 20)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)
            if not cap.isOpened():
                logger.warning(f"Camera {camera_id} reconnection failed")
                offline_frame = generate_offline_frame(camera_id)
                if frame_queue.qsize() > 450:
                    frame_queue.get()
                frame_queue.put(offline_frame)
            continue

        # Reset reconnect attempts on success
        reconnect_attempts = 0

        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.7)
        current_detections = []

        for result in results:
            if result.boxes is None:
                continue
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                embedding = extract_embedding(person_crop)
                tracked_embeddings[track_id] = embedding
                # logger.debug(f"Live embedding for track {track_id}: norm={np.linalg.norm(embedding):.2f}")

                current_detections.append({
                    'crop': person_crop,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'track_id': track_id,
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

        if frame_queue.qsize() > 450:
            frame_queue.get()
        frame_queue.put(cv2.resize(processed_frame, (854, 480)))

        if len(tracked_embeddings) > 100:
            oldest_track = min(tracked_embeddings.keys(), key=lambda k: tracked_embeddings[k].size)
            tracked_embeddings.pop(oldest_track)

    # Push offline frame if loop exits (camera permanently offline)
    offline_frame = generate_offline_frame(camera_id)
    if frame_queue.qsize() > 450:
        frame_queue.get()
    frame_queue.put(offline_frame)
    cap.release()
    logger.info(f"Camera {camera_id} process ended")

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    shared_namespace = manager.Namespace()
    shared_namespace.database_embeddings = None
    shared_namespace.database_names = []
    database = manager.dict(load_database(shared_namespace))  # Load into shared namespace
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"],
        'selected_cameras': manager.list()
    })

    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()

    cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
    cache_thread.start()

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
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        # logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    server_thread.join()

    # Main loop for keyboard interaction
    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #     elif key == ord('e'):
    #         database.clear()
    #         print("The database library has been cleaned up.")
    #     elif key == ord('n') and not shared_data['adding_new_person'].value:
    #         shared_data['adding_new_person'].value = True
    #         shared_data['current_name'] = get_person_name()
    #         if not shared_data['current_name']:
    #             shared_data['adding_new_person'].value = False
    #     elif key == ord('c'):
    #         shared_data['capture_signal'].value = True
    #     elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value:
    #         shared_data['dynamic_adding_new_person'].value = True
    #         shared_data['current_name'] = get_person_name()
    #         if not shared_data['current_name']:
    #             shared_data['dynamic_adding_new_person'].value = False

    # # Cleanup
    # for p in processes:
    #     p.terminate()
    #     p.join()










# =====================================================================================================================================












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
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Empty  # Explicitly import the correct Empty exception

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
logger = logging.getLogger(__name__)

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
    model = YOLO("yolo11n.engine", task='detect')
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
                         std=[0.229, 0.224, 0.225])
])

#########################################
#   GLOBAL VARIABLES                    #
#########################################

database_dir = './database'

# Global dictionary to store latest frames for each camera
latest_frames = {}

# Global variable to store frames for queue
frame_queues = []

# URLs of all IP cameras
camera_urls = [
    # "rtsp://192.168.1.168/stream_0",
    # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.116:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
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

def frame_consumer(camera_id, frame_queue, latest_frames):
    while True:
        try:
            frame = frame_queue.get(timeout=1)  # Get frame with timeout
            latest_frames[camera_id] = frame
            # logger.debug(f"Updated latest frame for Camera {camera_id}")
        except Empty:  # Catch the correct Empty exception
            # logger.debug(f"Queue empty for Camera {camera_id}")
            if camera_id not in latest_frames:
                latest_frames[camera_id] = generate_offline_frame(camera_id)
            time.sleep(0.1)  # Small delay to avoid busy-waiting

def extract_ip_from_url(url):
    """Extract the IP address from an RTSP URL."""
    match = re.search(r'@(\d+\.\d+\.\d+\.\d+):', url)
    return match.group(1) if match else "Unknown IP"

def reload_database(shared_namespace, database):
    """Reload database embeddings and names into shared_namespace."""
    try:
        _, embeddings_list, names_list = load_database(shared_namespace)
        if embeddings_list:
            shared_namespace.database_embeddings = torch.tensor(np.stack(embeddings_list))  # Update embeddings
            shared_namespace.database_names = names_list
            logger.info(f"Reloaded database: {len(names_list)} embeddings")
        else:
            shared_namespace.database_embeddings = None
            shared_namespace.database_names = []
            logger.warning("No embeddings loaded during reload")
    except Exception as e:
        logger.error(f"Error reloading database: {str(e)}")

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

def process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold=28):
    """
    Process dynamic capture from multiple cameras, collecting crops only from detected bounding boxes.
    Stops when the total number of captured crops reaches the threshold (default 28).
    """
    frame_count = 0
    max_frames = 200  # Fallback limit to prevent infinite loop
    while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
        crops = []
        for camera_id in cameras[:4]:  # Use the 4 selected cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]
                    crops.append(person_crop)
                # No else: only append crops when person is detected
            # No else: skip if frame is None
        
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
        cv2.putText(frame, str(track_id), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
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

# Main processing fucntion
def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database, shared_namespace):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)

    if not cap.isOpened():
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        # Push offline frame immediately if camera can't open
        offline_frame = generate_offline_frame(camera_id)
        frame_queue.put(offline_frame)
        return
    
    # logger.info(f"Camera {camera_id} initialized: {camera_url}")

    movement_trails = {}
    tracked_embeddings = {}
    reconnect_attempts = 0
    max_attempts = 5  # Limit reconnection attempts

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning(f"Camera {camera_id} failed to read frame, attempt {reconnect_attempts + 1}/{max_attempts}")
            reconnect_attempts += 1
            if reconnect_attempts >= max_attempts:
                logger.error(f"Camera {camera_id} offline after {max_attempts} attempts")
                offline_frame = generate_offline_frame(camera_id)
                if frame_queue.qsize() > 450:
                    frame_queue.get()
                frame_queue.put(offline_frame)
                break  # Exit loop if max attempts reached

            # Attempt to reconnect
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
            if not cap.isOpened():
                logger.warning(f"Camera {camera_id} reconnection failed")
                offline_frame = generate_offline_frame(camera_id)
                if frame_queue.qsize() > 450:
                    frame_queue.get()
                frame_queue.put(offline_frame)
            continue

        # Reset reconnect attempts on success
        reconnect_attempts = 0

        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
        current_detections = []

        for result in results:
            if result.boxes is None:
                continue
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                embedding = extract_embedding(person_crop)
                tracked_embeddings[track_id] = embedding
                # logger.debug(f"Live embedding for track {track_id}: norm={np.linalg.norm(embedding):.2f}")

                current_detections.append({
                    'crop': person_crop,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'track_id': track_id,
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

        if frame_queue.qsize() > 450:
            frame_queue.get()
        frame_queue.put(cv2.resize(processed_frame, (854, 480)))

        if len(tracked_embeddings) > 100:
            oldest_track = min(tracked_embeddings.keys(), key=lambda k: tracked_embeddings[k].size)
            tracked_embeddings.pop(oldest_track)

    # Push offline frame if loop exits (camera permanently offline)
    offline_frame = generate_offline_frame(camera_id)
    if frame_queue.qsize() > 450:
        frame_queue.get()
    frame_queue.put(offline_frame)
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
                          camera_data=camera_data)  # Add camera_data to the template

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


# @app.route('/add_person', methods=['POST'])
# @login_required
# def add_person():
#     data = request.json
#     type_ = data.get('type')
#     name = data.get('name')
#     cameras = data.get('cameras', [])
#     if not name or not type_:
#         return jsonify({"message": "Name and type are required"}), 400
    
#     if type_ == 'static':
#         shared_data['adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
        
#         # Fetch and crop the latest frames
#         crops = []
#         for camera_id in cameras[:4]:  # Limit to 4 cameras
#             frame = latest_frames.get(camera_id)
#             if frame is not None:
#                 # Detect and crop using YOLOv11
#                 results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
#                 if results and results[0].boxes is not None and len(results[0].boxes) > 0:
#                     # Select the most confident detection
#                     best_box = max(results[0].boxes, key=lambda b: b.conf)
#                     # Flatten the tensor and unpack
#                     x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
#                     person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box
#                     crops.append(person_crop)
#                 else:
#                     logger.warning(f"No person detected in camera {camera_id}")
#                     shared_data['adding_new_person'].value = False
#                     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
#             else:
#                 logger.warning(f"No frame available for camera {camera_id}")
#                 shared_data['adding_new_person'].value = False
#                 return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
        
#         if len(crops) == 4:  # Ensure we have all 4 views
#             success = process_new_person(crops, shared_data, database)
#             if success:
#                 return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
#             else:
#                 return jsonify({"message": "Failed to capture all views"}), 500
#         else:
#             shared_data['adding_new_person'].value = False
#             return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
#     elif type_ == 'dynamic':
#         shared_data['dynamic_adding_new_person'].value = True
#         shared_data['current_name'] = name
#         return jsonify({"message": f"Started dynamic addition for {name}"}), 200
    
#     return jsonify({"message": "Invalid addition type"}), 400

#temporary
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
        
        # Fetch and crop the latest frames
        crops = []
        for camera_id in cameras[:4]:  # Limit to 4 cameras
            frame = latest_frames.get(camera_id)
            if frame is not None:
                # Detect and crop using YOLOv11 (disabled strict check for testing)
                results = model.track(frame, stream=False, persist=True, classes=[0], conf=0.7)
                person_crop = frame  # Use full frame as fallback
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf)
                    x1, y1, x2, y2 = map(int, best_box.xyxy.squeeze().tolist())
                    person_crop = frame[y1:y2, x1:x2]  # Crop to bounding box if detected
                # else:  # Temporarily disabled for testing
                #     logger.warning(f"No person detected in camera {camera_id}")
                #     shared_data['adding_new_person'].value = False
                #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
                crops.append(person_crop)
            else:
                logger.warning(f"No frame available for camera {camera_id}")
                crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))  # Blank frame for testing
        
        # Proceed even if fewer than 4 crops (for testing)
        # if len(crops) == 4:  # Temporarily disabled
        success = process_new_person(crops, shared_data, database)
        if success:
            shared_data['database_update_event'].set()  # Signal update
            return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
        else:
            return jsonify({"message": "Failed to capture all views"}), 500
        # else:
        #     shared_data['adding_new_person'].value = False
        #     return jsonify({"message": "Failed to detect person in all cameras, please check and try again"}), 400
    
    elif type_ == 'dynamic':
        shared_data['dynamic_adding_new_person'].value = True
        shared_data['current_name'] = name
        shared_data['selected_cameras'] = cameras
        
        def run_dynamic_capture():
            success = process_dynamic_multi_camera(shared_data, database, latest_frames, cameras, threshold)
            if success:
                shared_data['database_update_event'].set()  # Signal update
            logger.info(f"Dynamic capture thread completed for {name}, success: {success}")
        
        threading.Thread(target=run_dynamic_capture, daemon=True).start()
        logger.info(f"Started dynamic capture thread for {name}")
        # return jsonify({"message": f"Started dynamic addition for {name}. Please wait until completion."}), 200
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
                          camera_data=camera_data)  # Add camera_data to the template

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


#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
    manager = mp.Manager()
    shared_namespace = manager.Namespace()
    shared_namespace.database_embeddings = None
    shared_namespace.database_names = []
    # database = manager.dict(load_database(shared_namespace))
    database_dict, embeddings_list, names_list = load_database(shared_namespace)
    database = manager.dict(database_dict)  # Only pass the dictionary
    shared_data = manager.dict({
        'dynamic_adding_new_person': manager.Value('b', False),
        'adding_new_person': manager.Value('b', False),
        'capture_signal': manager.Value('b', False),
        'capture_images': manager.list(),
        'current_view': manager.Value('i', 0),
        'current_name': "",
        'view_directions': ["Front", "Back", "Right", "Left"],
        'selected_cameras': manager.list(),
        'database_update_event': manager.Event()  # New event for signaling
    })

    rois = manager.list([[] for _ in camera_urls])
    latest_frames = manager.dict()

    cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
    cache_thread.start()

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
                             args=(i, frame_queues[i], latest_frames),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        # logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    server_thread.join()











# =====================================================================================================================================













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
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Empty  # Explicitly import the correct Empty exception

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
logger = logging.getLogger(__name__)

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
    model = YOLO("yolo11n.engine", task='detect')
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
                         std=[0.229, 0.224, 0.225])
])

#########################################
#   GLOBAL VARIABLES                    #
#########################################

database_dir = './database'

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
    "rtsp://admin:Lucky786$1@192.168.1.121:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.137:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.117:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
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
        cv2.putText(frame, str(track_id), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
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
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)

    if not cap.isOpened():
        logger.error(f"Camera {camera_id} failed to open: {camera_url}")
        offline_frame = generate_offline_frame(camera_id)
        frame_queue.put((offline_frame, offline_frame, []))  # Store raw, processed, and empty bboxes
        return
    
    logger.info(f"Camera {camera_id} initialized: {camera_url}")

    movement_trails = {}
    tracked_embeddings = {}
    reconnect_attempts = 0
    max_attempts = 5  # Limit reconnection attempts

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
                logger.warning(f"Camera {camera_id} reconnection failed")
                offline_frame = generate_offline_frame(camera_id)
                frame_queue.put((offline_frame, offline_frame, []))
            continue

        reconnect_attempts = 0

        # Save raw frame before processing
        raw_frame = frame.copy()

        # Perform YOLO detection
        results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
        current_detections = []
        bboxes = []

        for result in results:
            if result.boxes is None:
                continue
            track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))  # Store bounding box
                center = ((x1 + x2) // 2, y2)
                person_crop = frame[y1:y2, x1:x2]
                track_id = int(track_id) if track_id is not None else -1

                embedding = extract_embedding(person_crop)
                tracked_embeddings[track_id] = embedding
                # logger.debug(f"Live embedding for track {track_id}: norm={np.linalg.norm(embedding):.2f}")

                current_detections.append({
                    'crop': person_crop,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'track_id': track_id,
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

        # Store raw frame, processed frame, and bounding boxes
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

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
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

    initialize_queues()

    processes = []
    for i, url in enumerate(camera_urls):
        p = mp.Process(target=process_camera_feed,
                       args=(url, i, rois, frame_queues[i], shared_data, database, shared_namespace),
                       daemon=True)
        processes.append(p)
        p.start()
        logger.info(f"Started process for Camera {i}: {url}")

    # Start frame consumer threads
    consumer_threads = []
    for i in range(len(camera_urls)):
        t = threading.Thread(target=frame_consumer,
                             args=(i, frame_queues[i], latest_frames, latest_raw_frames, latest_detections),
                             daemon=True)
        consumer_threads.append(t)
        t.start()
        logger.info(f"Started frame consumer for Camera {i}")

    # Start Flask server
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
    server_thread.daemon = True
    server_thread.start()

    server_thread.join()












# =====================================================================================================================================






####################### STABLE VERSION -- WITHOUT FACE RECOGNITION (version-1) #######################







# #########################################
# #   INSTALL AND SET DEPENDENCIES        #
# #########################################

# import os
# # os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "debug" 
# import cv2
# import time
# import torch
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog
# import torch.multiprocessing as mp
# import logging
# import json
# import requests
# import shutil
# import re
# from io import BytesIO
# import threading
# from functools import wraps
# from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
# import urllib.parse
# from concurrent.futures import ThreadPoolExecutor
# from multiprocessing.queues import Empty  # Explicitly import the correct Empty exception

# logging.getLogger("ultralytics").setLevel(logging.WARNING)
# mp.set_start_method("spawn", force=True)

# # Configure logging
# # logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
# logger = logging.getLogger(__name__)

# # Flask app
# app = Flask(__name__)

# app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# # Simulated user database (replace with a real database in production)
# USERS = {"admin": "admin"}

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # print(f"=>> Device Used for Inference: {device}")

# # Initialize YOLO model
# try:
#     model = YOLO("yolo11n_12-2.engine", task='detect')
#     # logger.info("YOLO model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load YOLO model: {str(e)}")
#     raise

# telegram_pool = ThreadPoolExecutor(max_workers=5)

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_ibn_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function for crops
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# #########################################
# #   GLOBAL VARIABLES                    #
# #########################################

# database_dir = './body_database'

# # Global dictionary to store latest frames for each camera
# latest_frames = {}

# # Global dictionary to store latest raw frames and detections
# latest_raw_frames = {}  # Added for raw frames
# latest_detections = {}  # Added for detection results

# # Global variable to store frames for queue
# frame_queues = []

# # URLs of all IP cameras
# camera_urls = [
#     # "rtsp://192.168.1.168/stream_0",
#     # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
#     "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.121:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.137:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.117:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.99:554/Streaming/Channels/201?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.116:554/Streaming/Channels/101?transport=tcp",
#     # "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp",
# ]

# # Credentials for Telegram Channel/Group
# bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
# chat_id = '-1002599945824'

# # Global variable to cache active cameras
# active_cameras_cache = list(range(len(camera_urls)))  # Default to all cameras initially
# cache_lock = threading.Lock()

# # Track last alert time per track_id
# last_alert_time = {}  

# # Global variables
# database_embeddings = None  # Tensor of all embeddings
# database_names = []  # Corresponding names

# # Global variable to store processes (moved from main block to global scope)
# processes = []

# ##############################
# #   HELPER FUNCTIONS         #
# ##############################

# def add_person_to_database(name, images):
#     person_dir = os.path.join(database_dir, name)
#     os.makedirs(person_dir, exist_ok=True)
#     embeddings = []
#     for idx, image in enumerate(images):
#         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
#         cv2.imwrite(image_filename, image)
#         embeddings.append(extract_embedding(image))
#     return embeddings

# def extract_embedding(image):
#     try:
#         if image.size == 0:
#             logger.error("Empty image passed to extract_embedding")
#             return np.zeros(1000)  # Return dummy embedding if image is invalid
#         image_tensor = transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             embedding = reid_model(image_tensor)
#         embedding = embedding.squeeze(0).cpu().numpy()
#         # logger.debug(f"Embedding generated: shape={embedding.shape}, norm={np.linalg.norm(embedding):.2f}")
#         return embedding
#     except Exception as e:
#         logger.error(f"Error in extract_embedding: {str(e)}")
#         return np.zeros(1000)  # Fallback

# def match_embedding(query_embedding, database, shared_namespace, threshold=0.7):
#     if shared_namespace.database_embeddings is None or len(shared_namespace.database_embeddings) == 0:
#         logger.debug("Database embeddings empty or not initialized")
#         return None, 0
#     try:
#         # Move database_embeddings to GPU only when needed
#         db_embeddings = shared_namespace.database_embeddings.to(device)
#         query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0).to(device)
#         with torch.no_grad():
#             similarities = torch.cosine_similarity(query_tensor, db_embeddings)
#             best_similarity, best_idx = torch.max(similarities, dim=0)
#             best_similarity = best_similarity.item()
#             top_similarities = torch.topk(similarities, min(3, len(similarities))).values.tolist()
#             # logger.debug(f"Top similarities: {top_similarities}, Best: {best_similarity:.2f}, Threshold: {threshold}")
#             if best_similarity >= threshold:
#                 best_match = shared_namespace.database_names[best_idx]
#                 # logger.info(f"Match found: {best_match} with similarity {best_similarity:.2f}")
#                 return best_match, best_similarity
#         return None, 0
#     except Exception as e:
#         logger.error(f"Error in match_embedding: {str(e)}")
#         return None, 0

# def load_database(shared_namespace):
#     database = {}
#     embeddings_list = []
#     names_list = []
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 image_count = 0
#                 for image_filename in os.listdir(person_dir):
#                     image_path = os.path.join(person_dir, image_filename)
#                     image = cv2.imread(image_path)
#                     if image is not None:
#                         embedding = extract_embedding(image)
#                         embeddings.append(embedding)
#                         embeddings_list.append(embedding)
#                         names_list.append(person_name)
#                         image_count += 1
#                     else:
#                         logger.warning(f"Failed to load image: {image_path}")
#                 database[person_name] = embeddings
#                 # logger.info(f"Loaded {image_count} images for {person_name}")
#         if embeddings_list:
#             # Keep tensor on CPU to avoid CUDA serialization issues
#             shared_namespace.database_embeddings = torch.tensor(np.stack(embeddings_list))  # No .to(device) here
#             shared_namespace.database_names = names_list
#             # logger.info(f"Database loaded: {len(shared_namespace.database_names)} embeddings, {len(database)} unique persons")
#         else:
#             logger.warning("No embeddings loaded into database_embeddings")
#     else:
#         logger.error(f"Database directory {database_dir} does not exist")
#     return database, embeddings_list, names_list  # Return all three

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# def is_point_in_polygon(point, polygon):
#     """Return True if point is inside polygon, safely handling empty polygons."""
#     if not polygon or len(polygon) < 3:  # Minimum 3 points for a polygon
#         return False
#     return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0

# def draw_roi(frame):
#     roi_points = []
    
#     def mouse_callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             roi_points.append((x, y))
#             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#             if len(roi_points) > 1:
#                 cv2.line(frame, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
#             cv2.imshow("Draw ROI", frame)
        
#     cv2.imshow("Draw ROI", frame)
#     cv2.setMouseCallback("Draw ROI", mouse_callback)
    
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q') and len(roi_points) > 2:
#             cv2.line(frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
#             break
#     cv2.destroyWindow("Draw ROI")
#     return roi_points

# def initialize_queues():
#     global frame_queues
#     if not frame_queues:
#         print("Initializing frame queues...")
#         frame_queues.extend([mp.Queue(maxsize=2000) for _ in range(len(camera_urls))])
#         # print(f"Initialized {len(frame_queues)} frame queues.")
#         logger.info(f"Initialized {len(frame_queues)} frame queues.")

# def send_telegram_message(message):
#     url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
#     payload = {
#         "chat_id": chat_id,
#         "text": message
#     }
#     requests.post(url, data=payload)

# def send_photo_from_frame(frame, caption=None):
#     url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
#     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Lower quality for speed
#     photo_bytes = BytesIO(buffer.tobytes())
#     files = {"photo": ("alert.jpg", photo_bytes, "image/jpeg")}
#     payload = {"chat_id": chat_id, "caption": caption or ""}
#     try:
#         response = requests.post(url, data=payload, files=files, timeout=5)  # Add timeout
#         if not response.ok:
#             logger.error(f"Telegram send failed: {response.text}")
#     except requests.RequestException as e:
#         logger.error(f"Telegram send error: {str(e)}")

# def generate_offline_frame(camera_id, width=854, height=480):
#     # Create a blank image
#     offline_frame = np.zeros((height, width, 3), dtype=np.uint8)
#     # Add text indicating offline status
#     text = f"Camera {camera_id} offline"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     thickness = 2
#     text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
#     text_x = (width - text_size[0]) // 2
#     text_y = (height + text_size[1]) // 2
#     cv2.putText(offline_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
#     return offline_frame

# def update_active_cameras_periodically():
#     """Update the cached list of active cameras, first immediately, then periodically."""
#     global active_cameras_cache
#     while True:
#         temp_active_cameras = []
#         for i, url in enumerate(camera_urls):
#             cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#             if cap.isOpened():
#                 temp_active_cameras.append(i)
#                 cap.release()
#         with cache_lock:
#             active_cameras_cache = temp_active_cameras
#         logger.info(f"Updated active cameras cache: {active_cameras_cache}")
#         time.sleep(30)  # Update every 30 seconds

# def frame_consumer(camera_id, frame_queue, latest_frames, latest_raw_frames, latest_detections):
#     while True:
#         try:
#             # Expect tuple of (raw_frame, processed_frame, bboxes)
#             raw_frame, processed_frame, bboxes = frame_queue.get(timeout=1)
#             latest_frames[camera_id] = processed_frame
#             latest_raw_frames[camera_id] = raw_frame
#             latest_detections[camera_id] = bboxes
#             # logger.debug(f"Updated latest raw, processed frames, and detections for Camera {camera_id}")
#         except Empty:
#             # logger.debug(f"Queue empty for Camera {camera_id}")
#             if camera_id not in latest_frames:
#                 offline_frame = generate_offline_frame(camera_id)
#                 latest_frames[camera_id] = offline_frame
#                 latest_raw_frames[camera_id] = offline_frame
#                 latest_detections[camera_id] = []
#             time.sleep(0.1)

# def extract_ip_from_url(url):
#     """Extract the IP address from an RTSP URL."""
#     match = re.search(r'@(\d+\.\d+\.\d+\.\d+):', url)
#     return match.group(1) if match else "Unknown IP"

# def reload_database(shared_namespace, database):
#     """Reload database embeddings and names into shared_namespace."""
#     try:
#         _, embeddings_list, names_list = load_database(shared_namespace)
#         if embeddings_list:
#             shared_namespace.database_embeddings = torch.tensor(np.stack(embeddings_list))
#             shared_namespace.database_names = names_list
#             logger.info(f"Reloaded database: {len(names_list)} embeddings")
#         else:
#             shared_namespace.database_embeddings = None
#             shared_namespace.database_names = []
#             logger.warning("No embeddings loaded during reload")
#     except Exception as e:
#         logger.error(f"Error reloading database: {str(e)}")

# #########################################
# # Processing Functions for Each Mode    #
# #########################################

# def process_new_person(crops, shared_data, database):
#     """
#     Process crops from multiple cameras to capture all views (front, back, left, right) at once.
#     """
#     view_directions = shared_data['view_directions']  # ['Front', 'Back', 'Right', 'Left']
#     if len(crops) != len(view_directions):
#         logger.warning(f"Expected {len(view_directions)} crops, got {len(crops)}. Aborting.")
#         return False
    
#     # Capture all views at once
#     for i, crop in enumerate(crops):
#         view_direction = view_directions[i]
#         print(f"Captured {view_direction} view.")
#         shared_data['capture_images'].append(crop)

#     # Add the person to the database with all captured views
#     embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#     database[shared_data['current_name']] = embeddings
#     shared_data['adding_new_person'].value = False
#     shared_data['capture_images'][:] = []  # Clear the list
#     shared_data['current_view'].value = 0  # Reset view index
#     if 'new_person_track_id' in shared_data:
#         del shared_data['new_person_track_id']
#     return True

# def process_dynamic_multi_camera(shared_data, database, latest_raw_frames, latest_detections, cameras, threshold=28):
#     """
#     Process dynamic capture from multiple cameras, collecting crops only from detected bounding boxes.
#     Stops when the total number of captured crops reaches the threshold (default 28).
#     """
#     frame_count = 0
#     max_frames = 200  # Fallback limit to prevent infinite loop
#     while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
#         crops = []
#         for camera_id in cameras[:4]:  # Use the 4 selected cameras
#             raw_frame = latest_raw_frames.get(camera_id)
#             bboxes = latest_detections.get(camera_id, [])
#             if raw_frame is None or not bboxes:
#                 continue
            
#             # Use the most confident bounding box (assuming sorted by confidence in process_camera_feed)
#             x1, y1, x2, y2 = bboxes[0]
#             person_crop = raw_frame[y1:y2, x1:x2]
#             crops.append(person_crop)
        
#         frame_count += 1
#         if frame_count % 5 == 0 and crops:  # Capture every 5th frame if detections exist
#             for crop in crops:
#                 if len(shared_data['capture_images']) < threshold:  # Stop at threshold
#                     shared_data['capture_images'].append(crop)
#             logger.info(f"Dynamic capture: frame {frame_count}, captured {len(crops)} crops, total {len(shared_data['capture_images'])}/{threshold}")
        
#         time.sleep(0.05)  # Simulate frame rate (adjust as needed)

#     # Save to database when threshold is met or max frames reached with some crops
#     if shared_data['capture_images']:
#         embeddings = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#         database[shared_data['current_name']] = embeddings
#         logger.info(f"Dynamic capture completed: {len(shared_data['capture_images'])} crops saved for {shared_data['current_name']}")
#     else:
#         logger.warning(f"No crops captured for {shared_data['current_name']} after {max_frames} frames")
    
#     shared_data['dynamic_adding_new_person'].value = False
#     shared_data['capture_images'][:] = []
#     logger.info(f"Dynamic capture finished, is_active set to False for {shared_data['current_name']}")
#     return True

# def process_standard_detections(detections, frame, roi_points, movement_trails, database, camera_id, shared_namespace):
#     """
#     Batch process all standard detections (i.e. not adding a new person).
#     Compute ReID embeddings in batch and then update the display and movement trails.
#     """
#     if not detections:
#         return frame, movement_trails

#     for det in detections:
#         x1, y1, x2, y2 = det['bbox']
#         center = det['center']
#         track_id = det['track_id']
#         query_embedding = det['embedding']
#         match, similarity = match_embedding(query_embedding, database, shared_namespace)
#         label = match if match else "Unknown"

#         if is_point_in_polygon(center, roi_points):
#             # flabel = f"{label} ({similarity:.2f})"
#             flabel = f"{label}"
#             color = (0, 255, 0)
#         else:
#             # flabel = f"{label} ({similarity:.2f})"
#             flabel = f"{label}"
#             color = (0, 0, 255)

#         cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
#         cv2.putText(frame, str(track_id), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

#         if track_id not in movement_trails:
#             movement_trails[track_id] = []
#         movement_trails[track_id].append(center)
#         if len(movement_trails[track_id]) > 30:
#             movement_trails[track_id].pop(0)

#         for j in range(1, len(movement_trails[track_id])):
#             cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 1)

#         if len(movement_trails[track_id]) > 1:
#             last_point = movement_trails[track_id][-2]
#             current_point = movement_trails[track_id][-1]
#             entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
#             exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)
#             if entered_roi or exited_roi:
#                 current_time = time.time()
#                 if track_id not in last_alert_time or (current_time - last_alert_time[track_id]) > 10:
#                     direction = "entered" if entered_roi else "exited"
#                     message_alert = f"📷 Camera {camera_id} ALERT: Person {label} {direction} the ROI."
#                     telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
#                     last_alert_time[track_id] = current_time

#     return frame, movement_trails

# # Main processing function
# def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, database, shared_namespace):
#     # Enable OpenCV logging for FFmpeg
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
#     cap.set(cv2.CAP_PROP_FPS, 15)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)

#     if not cap.isOpened():
#         logger.error(f"Camera {camera_id} failed to open: {camera_url}. Check RTSP URL, FFmpeg support, or network.")
#         offline_frame = generate_offline_frame(camera_id)
#         frame_queue.put((offline_frame, offline_frame, []))
#         return
    
#     logger.info(f"Camera {camera_id} initialized: {camera_url}")

#     movement_trails = {}
#     tracked_embeddings = {}
#     reconnect_attempts = 0
#     max_attempts = 5

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             logger.warning(f"Camera {camera_id} failed to read frame, attempt {reconnect_attempts + 1}/{max_attempts}. POS_MSEC: {cap.get(cv2.CAP_PROP_POS_MSEC)}")
#             reconnect_attempts += 1
#             if reconnect_attempts >= max_attempts:
#                 logger.error(f"Camera {camera_id} offline after {max_attempts} attempts")
#                 offline_frame = generate_offline_frame(camera_id)
#                 frame_queue.put((offline_frame, offline_frame, []))
#                 break

#             time.sleep(1)
#             cap.release()
#             cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#             cap.set(cv2.CAP_PROP_FPS, 15)
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
#             if not cap.isOpened():
#                 logger.warning(f"Camera {camera_id} reconnection failed")
#                 offline_frame = generate_offline_frame(camera_id)
#                 frame_queue.put((offline_frame, offline_frame, []))
#             continue
#         reconnect_attempts = 0

#         # Save raw frame before processing
#         raw_frame = frame.copy()

#         # Perform YOLO detection
#         results = model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
#         current_detections = []
#         bboxes = []

#         for result in results:
#             if result.boxes is None:
#                 continue
#             track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
#             for box, track_id in zip(result.boxes, track_ids):
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 bboxes.append((x1, y1, x2, y2))  # Store bounding box
#                 center = ((x1 + x2) // 2, y2)
#                 person_crop = frame[y1:y2, x1:x2]
#                 track_id = int(track_id) if track_id is not None else -1

#                 embedding = extract_embedding(person_crop)
#                 tracked_embeddings[track_id] = embedding
#                 # logger.debug(f"Live embedding for track {track_id}: norm={np.linalg.norm(embedding):.2f}")

#                 current_detections.append({
#                     'crop': person_crop,
#                     'bbox': (x1, y1, x2, y2),
#                     'center': center,
#                     'track_id': track_id,
#                     'embedding': embedding
#                 })

#         roi_points = shared_rois[camera_id]
#         processed_frame = frame
#         if current_detections:
#             processed_frame, movement_trails = process_standard_detections(
#                 current_detections, processed_frame, roi_points, movement_trails, database, camera_id, shared_namespace
#             )

#         if roi_points and len(roi_points) >= 3:
#             cv2.polylines(processed_frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

#         # Check for database update event
#         if shared_data['database_update_event'].is_set():
#             logger.info(f"Camera {camera_id} received database update signal, reloading...")
#             reload_database(shared_namespace, database)
#             shared_data['database_update_event'].clear()  # Clear event after reload
#             logger.info(f"Camera {camera_id} database reloaded")

#         # Store raw frame, processed frame, and bounding boxes
#         if frame_queue.qsize() > 450:
#             frame_queue.get()
#         frame_queue.put((raw_frame, cv2.resize(processed_frame, (854, 480)), bboxes))

#         if len(tracked_embeddings) > 100:
#             oldest_track = min(tracked_embeddings.keys(), key=lambda k: tracked_embeddings[k].size)
#             tracked_embeddings.pop(oldest_track)

#     offline_frame = generate_offline_frame(camera_id)
#     frame_queue.put((offline_frame, offline_frame, []))
#     cap.release()
#     logger.info(f"Camera {camera_id} process ended")

# #########################################
# # FLASK FUNCTIONS                       #
# #########################################

# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if 'logged_in' not in session:
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     return decorated_function

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     """Handle login requests."""
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username in USERS and USERS[username] == password:
#             session['logged_in'] = True
#             return redirect(url_for('index', page=1))
#         else:
#             flash('Invalid username or password')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     """Handle logout requests."""
#     session.pop('logged_in', None)
#     return redirect(url_for('login'))

# @app.route('/', defaults={'page': 1})
# @app.route('/<int:page>')
# @login_required
# def index(page):
#     view_mode = request.args.get('view', 'separate')
#     cameras_per_page = 2 if view_mode == 'separate' else 6
#     total_cameras = len(camera_urls)
#     total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page
#     if page < 1:
#         page = 1
#     elif page > total_pages:
#         page = total_pages
#     start_idx = (page - 1) * cameras_per_page
#     end_idx = min(start_idx + cameras_per_page, total_cameras)
#     camera_indices = list(range(start_idx, end_idx))
    
#     # Create a dictionary mapping camera indices to their IPs
#     camera_data = {i: {"ip": extract_ip_from_url(camera_urls[i])} for i in range(len(camera_urls))}
    
#     persons = list(database.keys())
#     camera_count = len(camera_urls)
#     # Pass the cached active cameras to the template
#     with cache_lock:
#         active_cameras = active_cameras_cache[:]
    
#     logger.info(f"Rendering view_mode={view_mode}, page={page}, indices={camera_indices}, persons={persons}, camera_count={camera_count}, active_cameras={active_cameras}")
#     return render_template('index.html', 
#                           camera_indices=camera_indices, 
#                           current_page=page, 
#                           total_pages=total_pages,
#                           view_mode=view_mode,
#                           persons=persons,
#                           camera_count=camera_count,
#                           active_cameras=active_cameras,
#                           camera_data=camera_data)

# @app.route('/video_feed/<int:camera_id>')
# def video_feed(camera_id):
#     def generate():
#         while True:
#             frame = latest_frames.get(camera_id, generate_offline_frame(camera_id))
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_initial_frame/<int:camera_id>')
# @login_required
# def get_initial_frame(camera_id):
#     if camera_id >= len(camera_urls) or camera_id < 0:
#         return "Camera not found", 404
#     cap = cv2.VideoCapture(camera_urls[camera_id], cv2.CAP_FFMPEG)
#     success, frame = cap.read()
#     cap.release()
#     if not success:
#         return "Failed to capture frame", 500
#     ret, buffer = cv2.imencode('.jpg', frame)
#     return Response(buffer.tobytes(), mimetype='image/jpeg')

# @app.route('/save_roi/<int:camera_id>', methods=['POST'])
# @login_required
# def save_roi(camera_id):
#     if camera_id >= len(camera_urls) or camera_id < 0:
#         return jsonify({"error": "Invalid camera ID"}), 404
#     roi_data = request.json.get('roi', [])
#     roi_points = [(int(x), int(y)) for x, y in roi_data]
#     rois[camera_id] = roi_points  # Update shared list
#     print(f"Saved ROI for Camera {camera_id}: {rois[camera_id]}")
#     return jsonify({"status": "ROI saved successfully"}), 200

# @app.route('/remove_roi/<int:camera_id>', methods=['POST'])
# @login_required
# def remove_roi(camera_id):
#     if camera_id >= len(camera_urls) or camera_id < 0:
#         return jsonify({"error": "Invalid camera ID"}), 404
#     rois[camera_id] = []  # Clear the ROI points for this camera
#     logger.info(f"ROI removed for Camera {camera_id}")
#     return jsonify({"status": "ROI removed successfully"}), 200

# @app.route('/add_person', methods=['POST'])
# @login_required
# def add_person():
#     data = request.json
#     type_ = data.get('type')
#     name = data.get('name')
#     cameras = data.get('cameras', [])
#     disable_detection_check = data.get('disable_detection_check', False)  # From previous mod
#     threshold = data.get('threshold', 28)  # New parameter, default to 28
#     if not name or not type_:
#         return jsonify({"message": "Name and type are required"}), 400
    
#     if type_ == 'static':
#         shared_data['adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
        
#         # Fetch raw frames and use stored bounding boxes
#         crops = []
#         for camera_id in cameras[:4]:
#             raw_frame = latest_raw_frames.get(camera_id)
#             bboxes = latest_detections.get(camera_id, [])
#             if raw_frame is None:
#                 logger.warning(f"No raw frame available for camera {camera_id}")
#                 crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))
#                 continue
            
#             if bboxes and not disable_detection_check:
#                 x1, y1, x2, y2 = bboxes[0]  # Use most confident bounding box
#                 person_crop = raw_frame[y1:y2, x1:x2]
#                 crops.append(person_crop)
#             else:
#                 logger.warning(f"No detections for camera {camera_id}, using full frame")
#                 crops.append(raw_frame)
        
#         success = process_new_person(crops, shared_data, database)
#         if success:
#             shared_data['database_update_event'].set()
#             return jsonify({"message": f"Captured all views for {name} and added to database"}), 200
#         else:
#             shared_data['adding_new_person'].value = False
#             return jsonify({"message": "Failed to capture all views"}), 500
    
#     elif type_ == 'dynamic':
#         shared_data['dynamic_adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
        
#         def run_dynamic_capture():
#             success = process_dynamic_multi_camera(shared_data, database, latest_raw_frames, latest_detections, cameras, threshold)
#             if success:
#                 shared_data['database_update_event'].set()
#             logger.info(f"Dynamic capture thread completed for {name}, success: {success}")
        
#         threading.Thread(target=run_dynamic_capture, daemon=True).start()
#         logger.info(f"Started dynamic capture thread for {name}")
#         return '', 204
    
#     return jsonify({"message": "Invalid addition type"}), 400

# @app.route('/remove_person', methods=['POST'])
# @login_required
# def remove_person():
#     data = request.json
#     name = data.get('name')
#     if not name:
#         return jsonify({"message": "Name is required"}), 400
    
#     if name in database:
#         del database[name]
#         person_dir = os.path.join(database_dir, name)
#         if os.path.exists(person_dir):
#             shutil.rmtree(person_dir)
#         shared_data['database_update_event'].set()  # Signal update
#         return jsonify({"message": f"Removed {name} from database and deleted their images"}), 200
#     return jsonify({"message": f"{name} not found in database"}), 404

# @app.route('/clear_database', methods=['POST'])
# @login_required
# def clear_database():
#     database.clear()
#     # if os.path.exists(database_dir):
#     #     shutil.rmtree(database_dir)
#     # os.makedirs(database_dir, exist_ok=True)
#     shared_data['database_update_event'].set()  # Signal update
#     return jsonify({"message": "Database cleared successfully"}), 200

# @app.route('/get_person_list')
# @login_required
# def get_person_list():
#     try:
#         logger.info("Received request for /get_person_list")
#         persons = list(database.keys())
#         logger.info(f"Returning person list: {persons}")
#         return jsonify({"persons": persons})
#     except Exception as e:
#         logger.error(f"Error in get_person_list: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/get_active_cameras')
# @login_required
# def get_active_cameras():
#     """Return the cached list of active cameras (optional, kept for debugging)."""
#     with cache_lock:
#         return jsonify({"active_cameras": active_cameras_cache})

# @app.route('/capture_static', methods=['GET'])
# @login_required
# def capture_static():
#     name = request.args.get('name')
#     cameras = request.args.get('cameras')  # Comma-separated list of camera indices
#     if not name or not cameras:
#         flash("Name and cameras are required for capture.")
#         return redirect(url_for('index'))
    
#     try:
#         camera_indices = [int(i) for i in cameras.split(',')][:4]  # Limit to 4 cameras
#         if not all(0 <= i < len(camera_urls) for i in camera_indices):
#             raise ValueError("Invalid camera indices")
#     except ValueError:
#         flash("Invalid camera selection.")
#         return redirect(url_for('index'))
    
#     # Create camera_data with IPs for the selected camera indices
#     camera_data = {i: {"ip": extract_ip_from_url(camera_urls[i])} for i in camera_indices}
    
#     return render_template('capture_static.html', 
#                           name=name, 
#                           camera_indices=camera_indices,
#                           camera_data=camera_data)

# @app.route('/capture_dynamic', methods=['GET'])
# @login_required
# def capture_dynamic():
#     name = request.args.get('name')
#     cameras = request.args.get('cameras')
#     if not name or not cameras:
#         flash("Name and cameras are required for capture.")
#         return redirect(url_for('index'))
    
#     try:
#         camera_indices = [int(i) for i in cameras.split(',')][:4]  # Limit to 4 cameras
#         if not all(0 <= i < len(camera_urls) for i in camera_indices):
#             raise ValueError("Invalid camera indices")
#     except ValueError:
#         flash("Invalid camera selection.")
#         return redirect(url_for('index'))
    
#     camera_data = {i: {"ip": extract_ip_from_url(camera_urls[i])} for i in camera_indices}
    
#     return render_template('capture_dynamic.html', 
#                           name=name, 
#                           camera_indices=camera_indices,
#                           camera_data=camera_data)

# @app.route('/check_dynamic_status', methods=['GET'])
# @login_required
# def check_dynamic_status():
#     is_active = shared_data['dynamic_adding_new_person'].value
#     return jsonify({"is_active": is_active})

# @app.route('/get_dynamic_crop_count', methods=['GET'])
# @login_required
# def get_dynamic_crop_count():
#     crop_count = len(shared_data['capture_images'])
#     is_active = shared_data['dynamic_adding_new_person'].value
#     return jsonify({"crop_count": crop_count, "is_active": is_active})

# @app.route('/check_person_and_redirect', methods=['POST'])
# @login_required
# def check_person_and_redirect():
#     data = request.json
#     name = data.get('name')
#     cameras = data.get('cameras', [])
    
#     if not name or not cameras:
#         return jsonify({"message": "Name and cameras are required"}), 400
    
#     if name not in database:
#         return jsonify({"message": "Person not found in the database, use static addition method for new staff."}), 404
    
#     # Name exists, proceed to dynamic capture
#     camera_param = ','.join(map(str, cameras[:4]))  # Limit to 4 cameras
#     return jsonify({"redirect": f"/capture_dynamic?name={urllib.parse.quote(name)}&cameras={camera_param}"}), 200

# @app.route('/refresh_cameras', methods=['POST'])
# @login_required
# def refresh_cameras():
#     global processes
#     logger.info("Refreshing camera processes...")
#     restarted = 0
#     for i in range(len(camera_urls)):
#         p = processes[i]
#         if not p.is_alive():
#             logger.info(f"Restarting process for Camera {i}...")
#             p.terminate()
#             p.join()  # Ensure the old process is fully terminated
#             new_p = mp.Process(target=process_camera_feed,
#                               args=(camera_urls[i], i, rois, frame_queues[i], shared_data, database, shared_namespace),
#                               daemon=True)
#             new_p.start()
#             processes[i] = new_p
#             restarted += 1
#             logger.info(f"Restarted process for Camera {i}: {camera_urls[i]}")
#     return jsonify({"message": f"Refreshed camera streams. Restarted {restarted} processes."}), 200

# #########################################
# #   MAIN FUNCTION                       #
# #########################################

# if __name__ == "__main__":
#     manager = mp.Manager()
#     shared_namespace = manager.Namespace()
#     shared_namespace.database_embeddings = None
#     shared_namespace.database_names = []
#     database_dict, embeddings_list, names_list = load_database(shared_namespace)
#     database = manager.dict(database_dict)
#     shared_data = manager.dict({
#         'dynamic_adding_new_person': manager.Value('b', False),
#         'adding_new_person': manager.Value('b', False),
#         'capture_signal': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"],
#         'selected_cameras': manager.list(),
#         'database_update_event': manager.Event()
#     })

#     rois = manager.list([[] for _ in camera_urls])
#     latest_frames = manager.dict()
#     latest_raw_frames = manager.dict()  # Initialize raw frames
#     latest_detections = manager.dict()  # Initialize detections

#     cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
#     cache_thread.start()

#     initialize_queues()

#     processes = []
#     for i, url in enumerate(camera_urls):
#         p = mp.Process(target=process_camera_feed,
#                        args=(url, i, rois, frame_queues[i], shared_data, database, shared_namespace),
#                        daemon=True)
#         processes.append(p)
#         p.start()
#         # logger.info(f"Started process for Camera {i}: {url}")

#     # Start frame consumer threads
#     consumer_threads = []
#     for i in range(len(camera_urls)):
#         t = threading.Thread(target=frame_consumer,
#                              args=(i, frame_queues[i], latest_frames, latest_raw_frames, latest_detections),
#                              daemon=True)
#         consumer_threads.append(t)
#         t.start()
#         # logger.info(f"Started frame consumer for Camera {i}")

#     # Start Flask server
#     server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
#     server_thread.daemon = True
#     server_thread.start()

#     server_thread.join()













# ========================================================================================================





# Version-2 (with Face Detection)






# #########################################
# #   INSTALL AND SET DEPENDENCIES        #
# #########################################

# import re
# import os
# import cv2
# import time
# import torch
# import shutil
# import logging
# import requests
# import torchreid
# import threading
# import numpy as np
# import urllib.parse
# from io import BytesIO
# from functools import wraps
# from ultralytics import YOLO
# from torchvision import transforms
# import torch.multiprocessing as mp
# from tkinter import Tk, simpledialog
# from multiprocessing.queues import Empty
# from concurrent.futures import ThreadPoolExecutor
# from sklearn.metrics.pairwise import cosine_similarity
# from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify

# logging.getLogger("ultralytics").setLevel(logging.WARNING)
# mp.set_start_method("spawn", force=True)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'

# # Simulated user database
# USERS = {"admin": "admin"}

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Initialize models
# try:
#     body_model = YOLO("yolo11n_12-2.engine", task='detect')
#     # logger.info("Body YOLO model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load body YOLO model: {str(e)}")
#     raise

# try:
#     face_model = YOLO("yolov11n-face_12-2.engine", task='detect')  # Use a face detection model
#     # logger.info("Face YOLO model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load face YOLO model: {str(e)}")
#     raise

# torch.cuda.empty_cache()
# # Initialize ReID model for body
# reid_model = torchreid.models.build_model(
#     name='osnet_ibn_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing for ReID
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()
# ])

# # Preprocessing for face images
# face_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((160, 160)),  # Adjust based on face recognition model
#     transforms.ToTensor(),
# ])

# telegram_pool = ThreadPoolExecutor(max_workers=5)

# #########################################
# #   GLOBAL VARIABLES                    #
# #########################################

# database_dir = './body_database'
# face_database_dir = './face_database'

# # Global dictionaries
# latest_frames = {}
# latest_raw_frames = {}
# latest_detections = {}
# frame_queues = []

# # Camera URLs
# camera_urls = {
#     'body-cameras': [
#         "rtsp://admin:Lucky786$1@192.168.1.121:554/Streaming/Channels/101?transport=tcp",
#         "rtsp://admin:Lucky786$1@192.168.1.137:554/Streaming/Channels/101?transport=tcp",
#         "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp",
#         "rtsp://admin:Lucky786$1@192.168.1.117:554/Streaming/Channels/101?transport=tcp",
#         "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101?transport=tcp",
#     ],
#     'face-cameras': [
#         "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201?transport=tcp",
#     ]
# }

# # Flatten camera URLs for indexing
# all_camera_urls = camera_urls['body-cameras'] + camera_urls['face-cameras']
# camera_types = {i: 'body' for i in range(len(camera_urls['body-cameras']))}
# camera_types.update({i + len(camera_urls['body-cameras']): 'face' for i in range(len(camera_urls['face-cameras']))})

# # Telegram credentials
# bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
# chat_id = '-1002599945824'

# # Cache for active cameras
# active_cameras_cache = list(range(len(all_camera_urls)))
# cache_lock = threading.Lock()

# last_alert_time = {}
# body_database_embeddings = None
# body_database_names = []
# face_database = {}

# # Process list
# processes = []

# ##############################
# #   HELPER FUNCTIONS         #
# ##############################

# def add_body_to_database(name, images):
#     person_dir = os.path.join(database_dir, name)
#     os.makedirs(person_dir, exist_ok=True)
#     embeddings = []
#     for idx, image in enumerate(images):
#         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
#         cv2.imwrite(image_filename, image)
#         embeddings.append(extract_body_embedding(image))
#     return embeddings

# def add_face_to_database(name, images):
#     person_dir = os.path.join(face_database_dir, name)
#     os.makedirs(person_dir, exist_ok=True)
#     for idx, image in enumerate(images):
#         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
#         cv2.imwrite(image_filename, image)
#     return True

# def extract_body_embedding(image):
#     try:
#         if image.size == 0:
#             logger.error("Empty image for body embedding")
#             return np.zeros(1000)
#         image_tensor = transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             embedding = reid_model(image_tensor)
#         embedding = embedding.squeeze(0).cpu().numpy()
#         return embedding
#     except Exception as e:
#         logger.error(f"Error in extract_body_embedding: {str(e)}")
#         return np.zeros(1000)

# def extract_face_embedding(image):
#     try:
#         if image is None or image.size == 0:
#             logger.error("Empty or invalid image for face embedding")
#             return np.zeros(1000)  # Match ReID embedding size
#         # Use the same preprocessing as body embedding
#         image_tensor = transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             embedding = reid_model(image_tensor)
#         embedding = embedding.squeeze(0).cpu().numpy()
#         return embedding
#     except Exception as e:
#         logger.error(f"Error in extract_face_embedding: {str(e)}")
#         return np.zeros(1000)

# def match_body_embedding(query_embedding, database, shared_namespace, threshold=0.7):
#     if shared_namespace.body_database_embeddings is None or len(shared_namespace.body_database_embeddings) == 0:
#         return None, 0
#     try:
#         db_embeddings = shared_namespace.body_database_embeddings.to(device)
#         query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0).to(device)
#         with torch.no_grad():
#             similarities = torch.cosine_similarity(query_tensor, db_embeddings)
#             best_similarity, best_idx = torch.max(similarities, dim=0)
#             best_similarity = best_similarity.item()
#             if best_similarity >= threshold:
#                 return shared_namespace.body_database_names[best_idx], best_similarity
#         return None, 0
#     except Exception as e:
#         logger.error(f"Error in match_body_embedding: {str(e)}")
#         return None, 0

# def match_face_embedding(query_embedding, database, threshold=0.7):  # Adjust threshold if needed
#     if not database:
#         return None, 0
#     try:
#         max_similarity = 0
#         best_match = None
#         for name, embeddings in database.items():
#             for emb in embeddings:
#                 similarity = cosine_similarity([query_embedding], [emb])[0][0]
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     best_match = name
#         if max_similarity >= threshold:
#             return best_match, max_similarity
#         return None, 0
#     except Exception as e:
#         logger.error(f"Error in match_face_embedding: {str(e)}")
#         return None, 0

# def load_body_database(shared_namespace):
#     database = {}
#     embeddings_list = []
#     names_list = []
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image_path = os.path.join(person_dir, image_filename)
#                     image = cv2.imread(image_path)
#                     if image is not None:
#                         embedding = extract_body_embedding(image)
#                         embeddings.append(embedding)
#                         embeddings_list.append(embedding)
#                         names_list.append(person_name)
#                 database[person_name] = embeddings
#         if embeddings_list:
#             shared_namespace.body_database_embeddings = torch.tensor(np.stack(embeddings_list))
#             shared_namespace.body_database_names = names_list
#     return database, embeddings_list, names_list

# def load_face_database():
#     database = {}
#     if os.path.exists(face_database_dir):
#         for person_name in os.listdir(face_database_dir):
#             person_dir = os.path.join(face_database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image_path = os.path.join(person_dir, image_filename)
#                     image = cv2.imread(image_path)
#                     embedding = extract_face_embedding(image)
#                     if embedding is not None:
#                         embeddings.append(embedding)
#                 if embeddings:
#                     database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# def is_point_in_polygon(point, polygon):
#     if not polygon or len(polygon) < 3:
#         return False
#     return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0

# def draw_roi(frame):
#     roi_points = []
#     def mouse_callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             roi_points.append((x, y))
#             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#             if len(roi_points) > 1:
#                 cv2.line(frame, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
#             cv2.imshow("Draw ROI", frame)
#     cv2.imshow("Draw ROI", frame)
#     cv2.setMouseCallback("Draw ROI", mouse_callback)
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q') and len(roi_points) > 2:
#             cv2.line(frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
#             break
#     cv2.destroyWindow("Draw ROI")
#     return roi_points

# def initialize_queues():
#     global frame_queues
#     frame_queues.extend([mp.Queue(maxsize=2000) for _ in range(len(all_camera_urls))])
#     logger.info(f"Initialized {len(frame_queues)} frame queues.")

# def send_telegram_message(message):
#     url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
#     payload = {"chat_id": chat_id, "text": message}
#     requests.post(url, data=payload)

# def send_photo_from_frame(frame, caption=None):
#     url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
#     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
#     photo_bytes = BytesIO(buffer.tobytes())
#     files = {"photo": ("alert.jpg", photo_bytes, "image/jpeg")}
#     payload = {"chat_id": chat_id, "caption": caption or ""}
#     try:
#         response = requests.post(url, data=payload, files=files, timeout=5)
#         if not response.ok:
#             logger.error(f"Telegram send failed: {response.text}")
#     except requests.RequestException as e:
#         logger.error(f"Telegram send error: {str(e)}")

# def generate_offline_frame(camera_id, width=854, height=480):
#     offline_frame = np.zeros((height, width, 3), dtype=np.uint8)
#     text = f"Camera {camera_id} ({camera_types[camera_id]}) offline"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     thickness = 2
#     text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
#     text_x = (width - text_size[0]) // 2
#     text_y = (height + text_size[1]) // 2
#     cv2.putText(offline_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
#     return offline_frame

# def update_active_cameras_periodically():
#     global active_cameras_cache
#     while True:
#         temp_active_cameras = []
#         for i, url in enumerate(all_camera_urls):
#             cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#             if cap.isOpened():
#                 temp_active_cameras.append(i)
#                 cap.release()
#         with cache_lock:
#             active_cameras_cache = temp_active_cameras
#         logger.info(f"Updated active cameras cache: {active_cameras_cache}")
#         time.sleep(30)

# def frame_consumer(camera_id, frame_queue, latest_frames, latest_raw_frames, latest_detections):
#     while True:
#         try:
#             raw_frame, processed_frame, bboxes = frame_queue.get(timeout=1)
#             latest_frames[camera_id] = processed_frame
#             latest_raw_frames[camera_id] = raw_frame
#             latest_detections[camera_id] = bboxes
#         except Empty:
#             if camera_id not in latest_frames:
#                 offline_frame = generate_offline_frame(camera_id)
#                 latest_frames[camera_id] = offline_frame
#                 latest_raw_frames[camera_id] = offline_frame
#                 latest_detections[camera_id] = []
#             time.sleep(0.1)

# def extract_ip_from_url(url):
#     match = re.search(r'@(\d+\.\d+\.\d+\.\d+):', url)
#     return match.group(1) if match else "Unknown IP"

# def reload_databases(shared_namespace, body_database, face_database):
#     _, embeddings_list, names_list = load_body_database(shared_namespace)
#     if embeddings_list:
#         shared_namespace.body_database_embeddings = torch.tensor(np.stack(embeddings_list))
#         shared_namespace.body_database_names = names_list
#         logger.info(f"Reloaded body database: {len(names_list)} embeddings")
#     else:
#         shared_namespace.body_database_embeddings = None
#         shared_namespace.body_database_names = []
#     face_database.clear()
#     face_database.update(load_face_database())
#     logger.info(f"Reloaded face database: {len(face_database)} persons")

# #########################################
# # Processing Functions                  #
# #########################################

# def process_new_body_person(crops, shared_data, database):
#     view_directions = shared_data['view_directions']
#     if len(crops) != len(view_directions):
#         logger.warning(f"Expected {len(view_directions)} crops, got {len(crops)}")
#         return False
#     for i, crop in enumerate(crops):
#         shared_data['capture_images'].append(crop)
#     embeddings = add_body_to_database(shared_data['current_name'], shared_data['capture_images'])
#     database[shared_data['current_name']] = embeddings
#     shared_data['adding_new_person'].value = False
#     shared_data['capture_images'][:] = []
#     shared_data['current_view'].value = 0
#     return True

# def process_new_face_person(crops, shared_data, database):
#     if not crops:
#         logger.warning("No face crops provided")
#         return False
#     success = add_face_to_database(shared_data['current_name'], crops)
#     shared_data['adding_new_person'].value = False
#     shared_data['capture_images'][:] = []
#     return success

# def process_dynamic_multi_camera(shared_data, database, latest_raw_frames, latest_detections, cameras, threshold=28, camera_type='body'):
#     frame_count = 0
#     max_frames = 200
#     while (len(shared_data['capture_images']) < threshold and shared_data['dynamic_adding_new_person'].value):
#         crops = []
#         for camera_id in cameras[:4]:
#             raw_frame = latest_raw_frames.get(camera_id)
#             bboxes = latest_detections.get(camera_id, [])
#             if raw_frame is None or not bboxes:
#                 continue
#             x1, y1, x2, y2 = bboxes[0]
#             crop = raw_frame[y1:y2, x1:x2]
#             crops.append(crop)
#         frame_count += 1
#         if frame_count % 5 == 0 and crops:
#             for crop in crops:
#                 if len(shared_data['capture_images']) < threshold:
#                     shared_data['capture_images'].append(crop)
#         time.sleep(0.05)
#     if shared_data['capture_images']:
#         if camera_type == 'body':
#             embeddings = add_body_to_database(shared_data['current_name'], shared_data['capture_images'])
#             database[shared_data['current_name']] = embeddings
#         else:
#             add_face_to_database(shared_data['current_name'], shared_data['capture_images'])
#     shared_data['dynamic_adding_new_person'].value = False
#     shared_data['capture_images'][:] = []
#     return True

# def process_body_detections(detections, frame, roi_points, movement_trails, database, camera_id, shared_namespace):
#     if not detections:
#         return frame, movement_trails
#     for det in detections:
#         x1, y1, x2, y2 = det['bbox']
#         center = det['center']
#         track_id = det['track_id']
#         query_embedding = det['embedding']
#         match, similarity = match_body_embedding(query_embedding, database, shared_namespace)
#         label = match if match else "Unknown"
#         if is_point_in_polygon(center, roi_points):
#             flabel = f"{label}"
#             color = (0, 255, 0)
#         else:
#             flabel = f"{label}"
#             color = (0, 0, 255)
#         cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
#         cv2.putText(frame, str(track_id), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
#         if track_id not in movement_trails:
#             movement_trails[track_id] = []
#         movement_trails[track_id].append(center)
#         if len(movement_trails[track_id]) > 30:
#             movement_trails[track_id].pop(0)
#         for j in range(1, len(movement_trails[track_id])):
#             cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 1)
#         if len(movement_trails[track_id]) > 1:
#             last_point = movement_trails[track_id][-2]
#             current_point = movement_trails[track_id][-1]
#             entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
#             exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)
#             if entered_roi or exited_roi:
#                 current_time = time.time()
#                 if track_id not in last_alert_time or (current_time - last_alert_time[track_id]) > 10:
#                     direction = "entered" if entered_roi else "exited"
#                     message_alert = f"📷 Camera {camera_id} (Body) ALERT: Person {label} {direction} the ROI."
#                     telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
#                     last_alert_time[track_id] = current_time
#     return frame, movement_trails

# def process_face_detections(detections, frame, roi_points, movement_trails, database, camera_id):
#     if not detections:
#         return frame, movement_trails
#     for det in detections:
#         x1, y1, x2, y2 = det['bbox']
#         center = det['center']
#         track_id = det['track_id']
#         query_embedding = det['embedding']
#         match, similarity = match_face_embedding(query_embedding, database)
#         label = match if match else "Unknown"
#         if is_point_in_polygon(center, roi_points):
#             flabel = f"{label}"
#             color = (0, 255, 0)
#         else:
#             flabel = f"{label}"
#             color = (0, 0, 255)
#         cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
#         cv2.putText(frame, str(track_id), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
#         if track_id not in movement_trails:
#             movement_trails[track_id] = []
#         movement_trails[track_id].append(center)
#         if len(movement_trails[track_id]) > 30:
#             movement_trails[track_id].pop(0)
#         # for j in range(1, len(movement_trails[track_id])):
#         #     cv2.line(frame, movement_trails[track_id][j - 1], movement_trails[track_id][j], (255, 0, 0), 1)
#         if len(movement_trails[track_id]) > 1:
#             last_point = movement_trails[track_id][-2]
#             current_point = movement_trails[track_id][-1]
#             entered_roi = not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points)
#             exited_roi = is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points)
#             if entered_roi or exited_roi:
#                 current_time = time.time()
#                 if track_id not in last_alert_time or (current_time - last_alert_time[track_id]) > 10:
#                     direction = "entered" if entered_roi else "exited"
#                     message_alert = f"📷 Camera {camera_id} (Face) ALERT: Person {label} {direction} the ROI."
#                     telegram_pool.submit(send_photo_from_frame, frame.copy(), message_alert)
#                     last_alert_time[track_id] = current_time
#     return frame, movement_trails

# def process_camera_feed(camera_url, camera_id, shared_rois, frame_queue, shared_data, body_database, face_database, shared_namespace):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
#     cap.set(cv2.CAP_PROP_FPS, 15)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
#     if not cap.isOpened():
#         logger.error(f"Camera {camera_id} failed to open: {camera_url}")
#         offline_frame = generate_offline_frame(camera_id)
#         frame_queue.put((offline_frame, offline_frame, []))
#         return
#     logger.info(f"Camera {camera_id} ({camera_types[camera_id]}) initialized: {camera_url}")
#     movement_trails = {}
#     tracked_embeddings = {}
#     reconnect_attempts = 0
#     max_attempts = 5
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             logger.warning(f"Camera {camera_id} failed to read frame, attempt {reconnect_attempts + 1}/{max_attempts}")
#             reconnect_attempts += 1
#             if reconnect_attempts >= max_attempts:
#                 logger.error(f"Camera {camera_id} offline after {max_attempts} attempts")
#                 offline_frame = generate_offline_frame(camera_id)
#                 frame_queue.put((offline_frame, offline_frame, []))
#                 break
#             time.sleep(1)
#             cap.release()
#             cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#             cap.set(cv2.CAP_PROP_FPS, 15)
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
#             if not cap.isOpened():
#                 offline_frame = generate_offline_frame(camera_id)
#                 frame_queue.put((offline_frame, offline_frame, []))
#             continue
#         reconnect_attempts = 0
#         raw_frame = frame.copy()
#         current_detections = []
#         bboxes = []
#         camera_type = camera_types[camera_id]
#         if camera_type == 'body':
#             results = body_model.track(frame, stream=False, persist=True, device=device, classes=[0], conf=0.5)
#             for result in results:
#                 if result.boxes is None:
#                     continue
#                 track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
#                 for box, track_id in zip(result.boxes, track_ids):
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     bboxes.append((x1, y1, x2, y2))
#                     center = ((x1 + x2) // 2, y2)
#                     person_crop = frame[y1:y2, x1:x2]
#                     track_id = int(track_id) if track_id is not None else -1
#                     embedding = extract_body_embedding(person_crop)
#                     tracked_embeddings[track_id] = embedding
#                     current_detections.append({
#                         'crop': person_crop,
#                         'bbox': (x1, y1, x2, y2),
#                         'center': center,
#                         'track_id': track_id,
#                         'embedding': embedding
#                     })
#         else:
#             results = face_model.track(frame, stream=False, persist=True, device=device, conf=0.5)
#             for result in results:
#                 if result.boxes is None:
#                     continue
#                 track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)
#                 for box, track_id in zip(result.boxes, track_ids):
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     bboxes.append((x1, y1, x2, y2))
#                     center = ((x1 + x2) // 2, y2)
#                     face_crop = frame[y1:y2, x1:x2]
#                     track_id = int(track_id) if track_id is not None else -1
#                     embedding = extract_face_embedding(face_crop)
#                     if embedding is not None:
#                         tracked_embeddings[track_id] = embedding
#                         current_detections.append({
#                             'crop': face_crop,
#                             'bbox': (x1, y1, x2, y2),
#                             'center': center,
#                             'track_id': track_id,
#                             'embedding': embedding
#                         })
#         roi_points = shared_rois[camera_id]
#         processed_frame = frame
#         if current_detections:
#             if camera_type == 'body':
#                 processed_frame, movement_trails = process_body_detections(
#                     current_detections, processed_frame, roi_points, movement_trails, body_database, camera_id, shared_namespace
#                 )
#             else:
#                 processed_frame, movement_trails = process_face_detections(
#                     current_detections, processed_frame, roi_points, movement_trails, face_database, camera_id
#                 )
#         if roi_points and len(roi_points) >= 3:
#             cv2.polylines(processed_frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
#         if shared_data['database_update_event'].is_set():
#             logger.info(f"Camera {camera_id} reloading databases")
#             reload_databases(shared_namespace, body_database, face_database)
#             shared_data['database_update_event'].clear()
#         if frame_queue.qsize() > 450:
#             frame_queue.get()
#         frame_queue.put((raw_frame, cv2.resize(processed_frame, (854, 480)), bboxes))
#         if len(tracked_embeddings) > 100:
#             oldest_track = min(tracked_embeddings.keys(), key=lambda k: tracked_embeddings[k].size)
#             tracked_embeddings.pop(oldest_track)
#     offline_frame = generate_offline_frame(camera_id)
#     frame_queue.put((offline_frame, offline_frame, []))
#     cap.release()
#     logger.info(f"Camera {camera_id} process ended")

# #########################################
# # FLASK FUNCTIONS                       #
# #########################################

# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if 'logged_in' not in session:
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     return decorated_function

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username in USERS and USERS[username] == password:
#             session['logged_in'] = True
#             return redirect(url_for('index', page=1))
#         else:
#             flash('Invalid username or password')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.pop('logged_in', None)
#     return redirect(url_for('login'))

# @app.route('/', defaults={'page': 1})
# @app.route('/<int:page>')
# @login_required
# def index(page):
#     view_mode = request.args.get('view', 'separate')
#     cameras_per_page = 2 if view_mode == 'separate' else 6
#     total_cameras = len(all_camera_urls)
#     total_pages = (total_cameras + cameras_per_page - 1) // cameras_per_page
#     if page < 1:
#         page = 1
#     elif page > total_pages:
#         page = total_pages
#     start_idx = (page - 1) * cameras_per_page
#     end_idx = min(start_idx + cameras_per_page, total_cameras)
#     camera_indices = list(range(start_idx, end_idx))
#     camera_data = {i: {"ip": extract_ip_from_url(all_camera_urls[i]), "type": camera_types[i]} for i in range(len(all_camera_urls))}
#     persons = list(set(list(body_database.keys()) + list(face_database.keys())))
#     camera_count = len(all_camera_urls)
#     with cache_lock:
#         active_cameras = active_cameras_cache[:]
#     return render_template('index.html',
#                           camera_indices=camera_indices,
#                           current_page=page,
#                           total_pages=total_pages,
#                           view_mode=view_mode,
#                           persons=persons,
#                           camera_count=camera_count,
#                           active_cameras=active_cameras,
#                           camera_data=camera_data)

# @app.route('/video_feed/<int:camera_id>')
# def video_feed(camera_id):
#     def generate():
#         while True:
#             frame = latest_frames.get(camera_id, generate_offline_frame(camera_id))
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_initial_frame/<int:camera_id>')
# @login_required
# def get_initial_frame(camera_id):
#     if camera_id >= len(all_camera_urls) or camera_id < 0:
#         return "Camera not found", 404
#     cap = cv2.VideoCapture(all_camera_urls[camera_id], cv2.CAP_FFMPEG)
#     success, frame = cap.read()
#     cap.release()
#     if not success:
#         return "Failed to capture frame", 500
#     ret, buffer = cv2.imencode('.jpg', frame)
#     return Response(buffer.tobytes(), mimetype='image/jpeg')

# @app.route('/save_roi/<int:camera_id>', methods=['POST'])
# @login_required
# def save_roi(camera_id):
#     if camera_id >= len(all_camera_urls) or camera_id < 0:
#         return jsonify({"error": "Invalid camera ID"}), 404
#     roi_data = request.json.get('roi', [])
#     roi_points = [(int(x), int(y)) for x, y in roi_data]
#     rois[camera_id] = roi_points
#     return jsonify({"status": "ROI saved successfully"}), 200

# @app.route('/remove_roi/<int:camera_id>', methods=['POST'])
# @login_required
# def remove_roi(camera_id):
#     if camera_id >= len(all_camera_urls) or camera_id < 0:
#         return jsonify({"error": "Invalid camera ID"}), 404
#     rois[camera_id] = []
#     return jsonify({"status": "ROI removed successfully"}), 200

# @app.route('/add_person', methods=['POST'])
# @login_required
# def add_person():
#     data = request.json
#     type_ = data.get('type')
#     name = data.get('name')
#     cameras = data.get('cameras', [])
#     disable_detection_check = data.get('disable_detection_check', False)
#     threshold = data.get('threshold', 28)
#     detection_type = data.get('detection_type', 'body')  # 'body' or 'face'
#     if not name or not type_ or not detection_type:
#         return jsonify({"message": "Name, type, and detection_type are required"}), 400
#     if type_ == 'static':
#         shared_data['adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
#         crops = []
#         for camera_id in cameras[:4]:
#             raw_frame = latest_raw_frames.get(camera_id)
#             bboxes = latest_detections.get(camera_id, [])
#             if raw_frame is None:
#                 crops.append(np.zeros((720, 1280, 3), dtype=np.uint8))
#                 continue
#             if bboxes and not disable_detection_check:
#                 x1, y1, x2, y2 = bboxes[0]
#                 crop = raw_frame[y1:y2, x1:x2]
#                 crops.append(crop)
#             else:
#                 crops.append(raw_frame)
#         success = False
#         if detection_type == 'body':
#             success = process_new_body_person(crops, shared_data, body_database)
#         else:
#             success = process_new_face_person(crops, shared_data, face_database)
#         if success:
#             shared_data['database_update_event'].set()
#             return jsonify({"message": f"Captured for {name} ({detection_type}) and added to database"}), 200
#         else:
#             shared_data['adding_new_person'].value = False
#             return jsonify({"message": "Failed to capture"}), 500
#     elif type_ == 'dynamic':
#         shared_data['dynamic_adding_new_person'].value = True
#         shared_data['current_name'] = name
#         shared_data['selected_cameras'] = cameras
#         def run_dynamic_capture():
#             success = process_dynamic_multi_camera(shared_data, body_database if detection_type == 'body' else face_database,
#                                                   latest_raw_frames, latest_detections, cameras, threshold, detection_type)
#             if success:
#                 shared_data['database_update_event'].set()
#         threading.Thread(target=run_dynamic_capture, daemon=True).start()
#         return '', 204
#     return jsonify({"message": "Invalid addition type"}), 400

# @app.route('/remove_person', methods=['POST'])
# @login_required
# def remove_person():
#     data = request.json
#     name = data.get('name')
#     detection_type = data.get('detection_type', 'body')
#     if not name:
#         return jsonify({"message": "Name is required"}), 400
#     success = False
#     if detection_type == 'body' and name in body_database:
#         del body_database[name]
#         person_dir = os.path.join(database_dir, name)
#         if os.path.exists(person_dir):
#             shutil.rmtree(person_dir)
#         success = True
#     elif detection_type == 'face' and name in face_database:
#         del face_database[name]
#         person_dir = os.path.join(face_database_dir, name)
#         if os.path.exists(person_dir):
#             shutil.rmtree(person_dir)
#         success = True
#     if success:
#         shared_data['database_update_event'].set()
#         return jsonify({"message": f"Removed {name} from {detection_type} database"}), 200
#     return jsonify({"message": f"{name} not found in {detection_type} database"}), 404

# @app.route('/clear_database', methods=['POST'])
# @login_required
# def clear_database():
#     detection_type = request.json.get('detection_type', 'body')
#     if detection_type == 'body':
#         body_database.clear()
#         if os.path.exists(database_dir):
#             shutil.rmtree(database_dir)
#         os.makedirs(database_dir, exist_ok=True)
#     else:
#         face_database.clear()
#         if os.path.exists(face_database_dir):
#             shutil.rmtree(face_database_dir)
#         os.makedirs(face_database_dir, exist_ok=True)
#     shared_data['database_update_event'].set()
#     return jsonify({"message": f"{detection_type} database cleared successfully"}), 200

# @app.route('/get_person_list')
# @login_required
# def get_person_list():
#     detection_type = request.args.get('detection_type', 'body')
#     persons = list(body_database.keys()) if detection_type == 'body' else list(face_database.keys())
#     return jsonify({"persons": persons})

# @app.route('/get_active_cameras')
# @login_required
# def get_active_cameras():
#     with cache_lock:
#         return jsonify({"active_cameras": active_cameras_cache})

# @app.route('/capture_static', methods=['GET'])
# @login_required
# def capture_static():
#     name = request.args.get('name')
#     cameras = request.args.get('cameras')
#     detection_type = request.args.get('detection_type', 'body')
#     if not name or not cameras:
#         flash("Name and cameras are required for capture.")
#         return redirect(url_for('index'))
#     try:
#         camera_indices = [int(i) for i in cameras.split(',')][:4]
#         if not all(0 <= i < len(all_camera_urls) for i in camera_indices):
#             raise ValueError("Invalid camera indices")
#     except ValueError:
#         flash("Invalid camera selection.")
#         return redirect(url_for('index'))
#     camera_data = {i: {"ip": extract_ip_from_url(all_camera_urls[i]), "type": camera_types[i]} for i in camera_indices}
#     return render_template('capture_static.html',
#                           name=name,
#                           camera_indices=camera_indices,
#                           camera_data=camera_data,
#                           detection_type=detection_type)

# @app.route('/capture_dynamic', methods=['GET'])
# @login_required
# def capture_dynamic():
#     name = request.args.get('name')
#     cameras = request.args.get('cameras')
#     detection_type = request.args.get('detection_type', 'body')
#     if not name or not cameras:
#         flash("Name and cameras are required for capture.")
#         return redirect(url_for('index'))
#     try:
#         camera_indices = [int(i) for i in cameras.split(',')][:4]
#         if not all(0 <= i < len(all_camera_urls) for i in camera_indices):
#             raise ValueError("Invalid camera indices")
#     except ValueError:
#         flash("Invalid camera selection.")
#         return redirect(url_for('index'))
#     camera_data = {i: {"ip": extract_ip_from_url(all_camera_urls[i]), "type": camera_types[i]} for i in camera_indices}
#     return render_template('capture_dynamic.html',
#                           name=name,
#                           camera_indices=camera_indices,
#                           camera_data=camera_data,
#                           detection_type=detection_type)

# @app.route('/check_dynamic_status', methods=['GET'])
# @login_required
# def check_dynamic_status():
#     is_active = shared_data['dynamic_adding_new_person'].value
#     return jsonify({"is_active": is_active})

# @app.route('/get_dynamic_crop_count', methods=['GET'])
# @login_required
# def get_dynamic_crop_count():
#     crop_count = len(shared_data['capture_images'])
#     is_active = shared_data['dynamic_adding_new_person'].value
#     return jsonify({"crop_count": crop_count, "is_active": is_active})

# @app.route('/check_person_and_redirect', methods=['POST'])
# @login_required
# def check_person_and_redirect():
#     data = request.json
#     name = data.get('name')
#     cameras = data.get('cameras', [])
#     detection_type = data.get('detection_type', 'body')
#     if not name or not cameras:
#         return jsonify({"message": "Name and cameras are required"}), 400
#     database_to_check = body_database if detection_type == 'body' else face_database
#     if name not in database_to_check:
#         return jsonify({"message": f"Person not found in {detection_type} database, use static addition method."}), 404
#     camera_param = ','.join(map(str, cameras[:4]))
#     return jsonify({"redirect": f"/capture_dynamic?name={urllib.parse.quote(name)}&cameras={camera_param}&detection_type={detection_type}"}), 200

# @app.route('/refresh_cameras', methods=['POST'])
# @login_required
# def refresh_cameras():
#     global processes
#     restarted = 0
#     for i in range(len(all_camera_urls)):
#         p = processes[i]
#         if not p.is_alive():
#             p.terminate()
#             p.join()
#             new_p = mp.Process(target=process_camera_feed,
#                               args=(all_camera_urls[i], i, rois, frame_queues[i], shared_data, body_database, face_database, shared_namespace),
#                               daemon=True)
#             new_p.start()
#             processes[i] = new_p
#             restarted += 1
#     return jsonify({"message": f"Refreshed camera streams. Restarted {restarted} processes."}), 200

# #########################################
# #   MAIN FUNCTION                       #
# #########################################

# if __name__ == "__main__":
#     manager = mp.Manager()
#     shared_namespace = manager.Namespace()
#     shared_namespace.body_database_embeddings = None
#     shared_namespace.body_database_names = []
#     body_database_dict, embeddings_list, names_list = load_body_database(shared_namespace)
#     body_database = manager.dict(body_database_dict)
#     face_database = manager.dict(load_face_database())

#     shared_data = manager.dict({
#         'dynamic_adding_new_person': manager.Value('b', False),
#         'adding_new_person': manager.Value('b', False),
#         'capture_signal': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"],
#         'selected_cameras': manager.list(),
#         'database_update_event': manager.Event()
#     })
#     rois = manager.list([[] for _ in all_camera_urls])
#     latest_frames = manager.dict()
#     latest_raw_frames = manager.dict()
#     latest_detections = manager.dict()
#     cache_thread = threading.Thread(target=update_active_cameras_periodically, daemon=True)
#     cache_thread.start()
#     initialize_queues()
#     processes = []
#     for i, url in enumerate(all_camera_urls):
#         p = mp.Process(target=process_camera_feed,
#                        args=(url, i, rois, frame_queues[i], shared_data, body_database, face_database, shared_namespace),
#                        daemon=True)
#         processes.append(p)
#         p.start()
#     consumer_threads = []
#     for i in range(len(all_camera_urls)):
#         t = threading.Thread(target=frame_consumer,
#                              args=(i, frame_queues[i], latest_frames, latest_raw_frames, latest_detections),
#                              daemon=True)
#         consumer_threads.append(t)
#         t.start()
#     server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'threaded': True})
#     server_thread.daemon = True
#     server_thread.start()
#     server_thread.join()
















# -----------------------------------------------------------------------------------------------------------------









# STABLE VERSION with REID FACE DETECTION (VERSION-3)










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





















# ================================================================================================================================





















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
# os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "debug" 

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
logger = logging.getLogger(__name__)

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

#########################################
#   GLOBAL VARIABLES                    #
#########################################

database_dir = './body_database'

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
    "rtsp://admin:Lucky786$1@192.168.1.121:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.137:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.117:554/Streaming/Channels/101?transport=tcp", 
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
                break

            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 500)
            if not cap.isOpened():
                logger.warning(f"Camera {camera_id} reconnection failed")
                offline_frame = generate_offline_frame(camera_id)
                frame_queue.put((offline_frame, offline_frame, []))
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

        # Store raw frame, processed frame, and bounding boxes
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

#########################################
#   MAIN FUNCTION                       #
#########################################

if __name__ == "__main__":
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















# ------------------------------------------------------------------------------------------------------------------------------
















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
from datetime import datetime
from functools import wraps
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Empty  # Explicitly import the correct Empty exception
from torch.amp import autocast

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
logger = logging.getLogger(__name__)

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

#########################################
#   GLOBAL VARIABLES                    #
#########################################

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
    "rtsp://admin:Lucky786$1@192.168.1.121:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.137:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.117:554/Streaming/Channels/101?transport=tcp", 
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
    batch_duration = 300  # 30 minutes in seconds

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
















# ----------------------------------------------------------------------------------------------------------
















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

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
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
    "rtsp://admin:Lucky786$1@192.168.1.121:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.137:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.65:554/Streaming/Channels/101?transport=tcp", 
    "rtsp://admin:Lucky786$1@192.168.1.117:554/Streaming/Channels/101?transport=tcp", 
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





