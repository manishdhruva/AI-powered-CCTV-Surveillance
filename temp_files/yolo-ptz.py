# import cv2
# import os
# import time
# import torch
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from tkinter import Tk, simpledialog
# from onvif import ONVIFCamera
# from zeep.exceptions import Fault


# # Set up environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Initialize the ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# )
# reid_model.eval()

# # Preprocessing function for the ReID model
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize the YOLO model
# from ultralytics import YOLO
# model = YOLO("yolo11n.pt")

# # Database directory
# database_dir = './database'

# # ONVIF PTZ Control Setup
# camera_ip = '192.168.1.108'  # Replace with your camera IP
# camera_username = 'admin'  # Replace with your camera username
# camera_password = 'Lucky786$1'  # Replace with your camera password
# camera_port = 80

# # Connect to the ONVIF service
# camera = ONVIFCamera(camera_ip, camera_port, camera_username, camera_password, '/home/ultratechsystems/wsdl/')
# ptz_service = camera.create_ptz_service()
# media_service = camera.create_media_service()

# # ptz_config = ptz_service.GetConfigurations()
# # for config in ptz_config:
# #     print(f"Configuration Token: {config.token}")
# #     for attr in dir(config):
# #         print(attr)
# #     print('-'*80)

#     # print(f"Spaces: {config.Spaces}")

# try:
#     profiles = media_service.GetProfiles()
#     profile_token = profiles[0].token
# except Fault as e:
#     print(f"Authorization failed: {e}")


# # # PTZ Movement Function
# # def ptz_move(ptz, profile_token, pan, tilt, zoom, timeout=0.5):
# #     request = ptz.create_type('ContinuousMove')
# #     request.ProfileToken = profile_token
# #     request.Velocity = ptz.GetStatus({'ProfileToken': profile_token}).Position
# #     request.Velocity.PanTilt.x = pan  # Horizontal movement: -1 to 1
# #     request.Velocity.PanTilt.y = tilt # Vertical movement: -1 to 1
# #     request.Velocity.Zoom.x = zoom   # Zoom: -1 to 1
# #     ptz.ContinuousMove(request)
# #     time.sleep(timeout)
# #     ptz.Stop({'ProfileToken': profile_token})

# def ptz_move(ptz, profile_token, pan, tilt, zoom, timeout=0.5):
#     # Create ContinuousMove request
#     request = ptz.create_type('ContinuousMove')  
#     request.ProfileToken = profile_token

#     # Set the PanTilt and Zoom velocities directly
#     request.Velocity = ptz.create_type('PTZVelocity')

#     # Set the pan, tilt, and zoom velocities
#     request.Velocity.PanTilt = ptz.create_type('Vector2D')
#     request.Velocity.Zoom = ptz.create_type('Vector1D')

#     request.Velocity.PanTilt.x = pan  # Horizontal pan speed: -1 to 1
#     request.Velocity.PanTilt.y = tilt  # Vertical tilt speed: -1 to 1
#     request.Velocity.Zoom.x = zoom  # Zoom speed: -1 to 1

#     # Set the spaces for pan/tilt and zoom
#     request.Velocity.PanTilt.space = 'http://www.onvif.org/ver20/ptz/wsdl/DefaultContinuousPanTiltVelocitySpace'
#     request.Velocity.Zoom.space = 'http://www.onvif.org/ver20/ptz/wsdl/DefaultContinuousZoomVelocitySpace'

#     # Perform the continuous move command
#     ptz.ContinuousMove(request)
    
#     # Sleep for the timeout duration before stopping
#     time.sleep(timeout)
    
#     # Stop the camera after the timeout
#     ptz.Stop({'ProfileToken': profile_token})


# # PTZ Center Tracking Function
# def ptz_track_person(bbox_center, frame_center):
#     pan_speed = (bbox_center[0] - frame_center[0]) / frame_center[0]  # Normalize to -1 to 1
#     tilt_speed = (bbox_center[1] - frame_center[1]) / frame_center[1]  # Normalize to -1 to 1
#     ptz_move(ptz_service, profile_token, pan=pan_speed, tilt=tilt_speed, zoom=0.0)

# # Function to extract embeddings
# def extract_embedding(image, model, transform):
#     image_tensor = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         embedding = model(image_tensor)
#     return embedding.squeeze(0).numpy()

# # Function to match query embeddings with the database
# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match = None
#     best_similarity = 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(
#                 query_embedding.reshape(1, -1),
#                 embedding.reshape(1, -1)
#             )[0][0]
#             if similarity > best_similarity and similarity > threshold:
#                 best_match = name
#                 best_similarity = similarity
#     return best_match, best_similarity

# # Load existing database
# def load_database(database_dir, model, transform):
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image_path = os.path.join(person_dir, image_filename)
#                     image = cv2.imread(image_path)
#                     if image is not None:
#                         embeddings.append(extract_embedding(image, model, transform))
#                 database[person_name] = embeddings
#     return database

# database = load_database(database_dir, reid_model, transform)

# # Initialize webcam
# ip_camera_url = "rtsp://admin:Lucky786$1@192.168.1.108:554/Streaming/Channels/101"
# cap = cv2.VideoCapture(ip_camera_url)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# print("Press 'n' to add a new person, 'q' to quit.")

# frame_count = 0

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Failed to read from camera.")
#         break

#     frame_count += 1
#     if frame_count % 2 != 0:
#         continue

#     # Frame dimensions
#     frame_height, frame_width = frame.shape[:2]
#     frame_center = (frame_width // 2, frame_height // 2)

#     # Run YOLO inference
#     results = model.track(frame, stream=True, classes=[0], persist=True)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             person_crop = frame[y1:y2, x1:x2]

#             # Extract embedding
#             query_embedding = extract_embedding(person_crop, reid_model, transform)
#             match, similarity = match_embedding(query_embedding, database)
#             label = match if match else "Unknown"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

#             # Track person with PTZ
#             bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
#             ptz_track_person(bbox_center, frame_center)

#     frame_dis = cv2.resize(frame, (960, 540))
#     cv2.imshow("Webcam", frame_dis)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





#--------------------------------------------------------------------------------




# import sys
# from onvif import ONVIFCamera

# mycam = ONVIFCamera('192.168.1.108', 80, 'admin', 'Lucky786$1', )
# media = mycam.create_media_service()
# ptz = mycam.create_ptz_service()
# media_profile = media.GetProfiles()[0]

# moverequest = ptz.create_type('AbsoluteMove')
# moverequest.ProfileToken = media_profile.token
# moverequest.Position=ptz.GetStatus({'ProfileToken': media_profile.token}).Position

# try: 
#     moverequest.Position.PanTilt.x = float(sys.argv[1])
#     moverequest.Position.PanTilt.y = float(sys.argv[2])
#     moverequest.Position.Zoom.x = float(sys.argv[3])
# except:
#     pass
    
# ptz.AbsoluteMove(moverequest)













# import cv2
# from onvif import ONVIFCamera
# from ultralytics import YOLO  # Ensure you are using the ultralytics library for YOLOv8
# import os

# # Set up environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Connect to the ONVIF camera
# camera_ip = '192.168.1.108'
# camera_port = 80
# camera_user = 'admin'
# camera_password = 'Lucky786$1'

# mycam = ONVIFCamera(camera_ip, camera_port, camera_user, camera_password)
# media = mycam.create_media_service()
# ptz = mycam.create_ptz_service()
# media_profile = media.GetProfiles()[0]

# # ptz_config = ptz.GetConfigurationOptions({'ConfigurationToken': media_profile.PTZConfiguration.token})
# # print(ptz_config)
# # exit(0)

# # Load YOLO model
# model = YOLO('yolo11n.pt')  # Replace with your YOLO model path (YOLOv8 is recommended)

# # Open the video stream
# cap = cv2.VideoCapture(f"rtsp://{camera_user}:{camera_password}@{camera_ip}/stream")

# # def move_camera(ptz, profile_token, pan_delta, tilt_delta):
# #     # Update the camera position based on the pan and tilt deltas
# #     request = ptz.create_type('ContinuousMove')
# #     request.ProfileToken = profile_token
# #     velocity = ptz.GetStatus({'ProfileToken': profile_token}).Position

# #     if pan_delta != 0:
# #         velocity.PanTilt.x += pan_delta
# #     if tilt_delta != 0:
# #         velocity.PanTilt.y += tilt_delta

# #     request.Velocity = velocity
# #     ptz.ContinuousMove(request)
# #     ptz.Stop({'ProfileToken': profile_token})  # Stop after movement


# def move_camera(ptz, profile_token, pan_delta, tilt_delta, zoom_delta=None):
#     # Clamp values to be within the supported range
#     pan_delta = max(-1.0, min(1.0, pan_delta))
#     tilt_delta = max(-1.0, min(1.0, tilt_delta))
#     if zoom_delta is not None:
#         zoom_delta = max(-1.0, min(1.0, zoom_delta))

#     # Create RelativeMove request
#     request = ptz.create_type('RelativeMove')
#     request.ProfileToken = profile_token

#     # Set translation values
#     request.Translation = {
#         'PanTilt': {'x': pan_delta, 'y': tilt_delta},
#         'Zoom': {'x': zoom_delta} if zoom_delta is not None else None
#     }

#     try:
#         ptz.RelativeMove(request)
#     except Exception as e:
#         print(f"Error during PTZ move: {e}")


# def continuous_move(ptz, profile_token, pan_speed, tilt_speed, zoom_speed=None):
#     # Clamp speed values
#     pan_speed = max(-1.0, min(1.0, pan_speed))
#     tilt_speed = max(-1.0, min(1.0, tilt_speed))
#     if zoom_speed is not None:
#         zoom_speed = max(-1.0, min(1.0, zoom_speed))

#     # Create ContinuousMove request
#     request = ptz.create_type('ContinuousMove')
#     request.ProfileToken = profile_token
#     request.Velocity = {
#         'PanTilt': {'x': pan_speed, 'y': tilt_speed},
#         'Zoom': {'x': zoom_speed} if zoom_speed is not None else None
#     }

#     try:
#         ptz.ContinuousMove(request)
#     except Exception as e:
#         print(f"Error during continuous move: {e}")


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO model on the frame
#     results = model.track(frame, classes=[0])

#     # Parse results
#     for result in results:
#         boxes = result.boxes  # Detected bounding boxes
#         for box in boxes:
#             cls = int(box.cls)  # Class ID
#             if cls == 0:  # Class ID 0 corresponds to "person"
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Bounding box center
#                 frame_h, frame_w, _ = frame.shape
#                 frame_center_x, frame_center_y = frame_w // 2, frame_h // 2

#                 # Calculate offsets
#                 offset_x = cx - frame_center_x
#                 offset_y = cy - frame_center_y

#                 # Map offsets to PTZ control values
#                 pan_delta = -offset_x / frame_w  # Adjust these scalars as needed
#                 tilt_delta = -offset_y / frame_h

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#                 # Move the camera
#                 # move_camera(ptz, media_profile.token, pan_delta, tilt_delta)
#                 continuous_move(ptz, media_profile.token, pan_delta, tilt_delta)
#                 break

#     # Display the frame for debugging
#     cv2.imshow("Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







# import cv2
# import time
# from onvif import ONVIFCamera
# from ultralytics import YOLO

# # Initialize YOLO model
# yolo_model = YOLO('yolov8n.pt')  # Load a YOLO model (adjust model path as needed)

# # Initialize PTZ camera
# camera_ip = '192.168.1.108'
# camera_port = 80
# camera_user = 'admin'
# camera_password = 'Lucky786$1'
# mycam = ONVIFCamera(camera_ip, camera_port, camera_user, camera_password)

# # Create media and PTZ services
# media = mycam.create_media_service()
# ptz = mycam.create_ptz_service()
# media_profile = media.GetProfiles()[0]

# # # Function to move the camera
# # def move_camera(ptz, profile_token, pan_delta, tilt_delta):
# #     request = ptz.create_type('ContinuousMove')
# #     request.ProfileToken = profile_token

# #     # Set PTZ velocity based on deltas
# #     request.Velocity = {
# #         'PanTilt': {'x': pan_delta, 'y': tilt_delta},
# #         'Zoom': {'x': 0.0}
# #     }
# #     request.Timeout = 'PT1S'  # 1-second timeout

# #     try:
# #         ptz.ContinuousMove(request)
# #     except Exception as e:
# #         print(f"Error moving camera: {e}")

# # # Function to track a person
# # def track_person(ptz, profile_token, bbox, frame_width, frame_height, previous_offsets):
# #     # Calculate bounding box center
# #     bbox_center_x = (bbox[0] + bbox[2]) / 2
# #     bbox_center_y = (bbox[1] + bbox[3]) / 2

# #     # Calculate frame center
# #     frame_center_x = frame_width / 2
# #     frame_center_y = frame_height / 2

# #     # Calculate offset from frame center
# #     offset_x = bbox_center_x - frame_center_x
# #     offset_y = bbox_center_y - frame_center_y

# #     # Normalize offsets to PTZ range
# #     pan_delta = offset_x / frame_width
# #     tilt_delta = offset_y / frame_height

# #     # Clamp movement to avoid overshooting
# #     pan_delta = max(-0.1, min(0.1, pan_delta))
# #     tilt_delta = max(-0.1, min(0.1, tilt_delta))

# #     # Smooth movement using exponential smoothing
# #     smoothed_pan_delta = 0.8 * previous_offsets['pan'] + 0.2 * pan_delta
# #     smoothed_tilt_delta = 0.8 * previous_offsets['tilt'] + 0.2 * tilt_delta

# #     # Update previous offsets
# #     previous_offsets['pan'] = smoothed_pan_delta
# #     previous_offsets['tilt'] = smoothed_tilt_delta

# #     # Check if movement is significant
# #     if abs(offset_x) < 0.05 * frame_width and abs(offset_y) < 0.05 * frame_height:
# #         return  # No significant movement needed

# #     # Move the camera
# #     move_camera(ptz, profile_token, smoothed_pan_delta, smoothed_tilt_delta)

# # Function to move the camera with configurable speeds
# def move_camera(ptz, profile_token, pan_speed, tilt_speed):
#     request = ptz.create_type('ContinuousMove')
#     request.ProfileToken = profile_token

#     # Set PTZ velocity based on speeds
#     request.Velocity = {
#         'PanTilt': {'x': pan_speed, 'y': tilt_speed},
#         'Zoom': {'x': 0.0}  # No zoom adjustments for now
#     }
#     request.Timeout = 'PT1S'  # 1-second timeout

#     try:
#         ptz.ContinuousMove(request)
#     except Exception as e:
#         print(f"Error moving camera: {e}")

# # Function to track a person with faster response
# def track_person(ptz, profile_token, bbox, frame_width, frame_height, previous_offsets):
#     # Calculate bounding box center
#     bbox_center_x = (bbox[0] + bbox[2]) / 2
#     bbox_center_y = (bbox[1] + bbox[3]) / 2

#     # Calculate frame center
#     frame_center_x = frame_width / 2
#     frame_center_y = frame_height / 2

#     # Calculate offset from frame center
#     offset_x = bbox_center_x - frame_center_x
#     offset_y = bbox_center_y - frame_center_y

#     # Normalize offsets to PTZ velocity range (-1.0 to 1.0)
#     pan_speed = offset_x / frame_width * 2  # Scale for faster pan
#     tilt_speed = offset_y / frame_height * 2  # Scale for faster tilt

#     # Clamp speed to camera's allowable range
#     pan_speed = max(-1.0, min(1.0, pan_speed))
#     tilt_speed = max(-1.0, min(1.0, tilt_speed))

#     # Apply exponential smoothing with faster responsiveness
#     smoothed_pan_speed = 0.5 * previous_offsets['pan'] + 0.5 * pan_speed
#     smoothed_tilt_speed = 0.5 * previous_offsets['tilt'] + 0.5 * tilt_speed

#     # Update previous offsets
#     previous_offsets['pan'] = smoothed_pan_speed
#     previous_offsets['tilt'] = smoothed_tilt_speed

#     # Move the camera with smoothed speed
#     move_camera(ptz, profile_token, smoothed_pan_speed, smoothed_tilt_speed)


# # Main function
# def main():
#     # Open video stream or camera feed
#     ip_camera_url = "rtsp://admin:Lucky786$1@192.168.1.108:554/Streaming/Channels/101"
#     cap = cv2.VideoCapture(ip_camera_url)  # Use 0 for webcam or provide a video file path

#     # Get frame dimensions
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture video frame")
#         return

#     frame_height, frame_width = frame.shape[:2]

#     # Initialize previous offsets for smoothing
#     previous_offsets = {'pan': 0.0, 'tilt': 0.0}

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture video frame")
#             break

#         # Run YOLO detection
#         results = yolo_model(frame, classes=[0])
#         if not results[0].boxes:
#             print("No person detected, stopping PTZ movement")
#             ptz.Stop({'ProfileToken': media_profile.token})
#             continue

#         # Take the first detected person's bounding box
#         bbox = results[0].boxes[0].xyxy[0].cpu().numpy()  # Bounding box (x1, y1, x2, y2)

#         # Track the person
#         track_person(ptz, media_profile.token, bbox, frame_width, frame_height, previous_offsets)

#         # Visualize bounding box and frame center for debugging
#         x1, y1, x2, y2 = map(int, bbox)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.circle(frame, (frame_width // 2, frame_height // 2), 5, (0, 0, 255), -1)
#         cv2.imshow('PTZ Tracking', frame)

#         # Break loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

# # Run main function
# if __name__ == '__main__':
#     main()






import cv2
import time
from onvif import ONVIFCamera
from ultralytics import YOLO

# Initialize YOLO model
yolo_model = YOLO('yolo11n.pt')  # Load a YOLO model (adjust model path as needed)

# Initialize PTZ camera
camera_ip = '192.168.1.108'
camera_port = 80
camera_user = 'admin'
camera_password = 'Lucky786$1'
mycam = ONVIFCamera(camera_ip, camera_port, camera_user, camera_password)

# Create media and PTZ services
media = mycam.create_media_service()
ptz = mycam.create_ptz_service()
media_profile = media.GetProfiles()[0]

# Function to get the current PTZ position (Pan/Tilt)
def get_current_position(ptz, profile_token):
    try:
        status = ptz.GetStatus({'ProfileToken': profile_token})
        if status.Position:
            return status.Position.PanTilt.x, status.Position.PanTilt.y
    except Exception as e:
        print(f"Error fetching current position: {e}")
    return 0.0, 0.0  # Default position if fetching fails

# Function to move the camera
def move_camera(ptz, profile_token, pan_speed, tilt_speed):
    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = profile_token

    # Set PTZ velocity based on speeds
    request.Velocity = {
        'PanTilt': {'x': pan_speed, 'y': tilt_speed},
        'Zoom': {'x': 0.0}  # No zoom adjustments for now
    }
    request.Timeout = 'PT1S'  # 1-second timeout

    try:
        ptz.ContinuousMove(request)
    except Exception as e:
        print(f"Error moving camera: {e}")

# Function to track a person with faster response
def track_person(ptz, profile_token, bbox, frame_width, frame_height, previous_offsets):
    # Calculate bounding box center
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # Calculate frame center
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    # Calculate offset from frame center
    offset_x = bbox_center_x - frame_center_x
    offset_y = bbox_center_y - frame_center_y

    # Normalize offsets to PTZ velocity range (-1.0 to 1.0)
    pan_speed = offset_x / frame_width * 2  # Scale for faster pan
    tilt_speed = offset_y / frame_height * 2  # Scale for faster tilt

    # Clamp speed to camera's allowable range
    pan_speed = max(-1.0, min(1.0, pan_speed))
    tilt_speed = max(-1.0, min(1.0, tilt_speed))

    # Apply exponential smoothing with faster responsiveness
    smoothed_pan_speed = 0.5 * previous_offsets['pan'] + 0.5 * pan_speed
    smoothed_tilt_speed = 0.5 * previous_offsets['tilt'] + 0.5 * tilt_speed

    # Update previous offsets
    previous_offsets['pan'] = smoothed_pan_speed
    previous_offsets['tilt'] = smoothed_tilt_speed

    # Move the camera with smoothed speed
    move_camera(ptz, profile_token, smoothed_pan_speed, smoothed_tilt_speed)


# Main function
def main():
    # Open video stream or camera feed
    ip_camera_url = "rtsp://admin:Lucky786$1@192.168.1.108:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(ip_camera_url)  # Use 0 for webcam or provide a video file path

    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame")
        return

    frame_height, frame_width = frame.shape[:2]

    # Initialize previous offsets for smoothing
    previous_offsets = {'pan': 0.0, 'tilt': 0.0}

    # Get the initial position of the camera
    current_pan, current_tilt = get_current_position(ptz, media_profile.token)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame")
            break

        # Run YOLO detection
        results = yolo_model(frame, classes=[0])
        if not results[0].boxes:
            print("No person detected, stopping PTZ movement")
            ptz.Stop({'ProfileToken': media_profile.token})
            continue

        # Take the first detected person's bounding box
        bbox = results[0].boxes[0].xyxy[0].cpu().numpy()  # Bounding box (x1, y1, x2, y2)

        # Track the person
        track_person(ptz, media_profile.token, bbox, frame_width, frame_height, previous_offsets)

        # Visualize bounding box and frame center for debugging
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (frame_width // 2, frame_height // 2), 5, (0, 0, 255), -1)
        cv2.imshow('PTZ Tracking', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run main function
if __name__ == '__main__':
    main()
