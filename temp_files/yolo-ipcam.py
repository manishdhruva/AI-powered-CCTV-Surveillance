# # import os
# # os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # from ultralytics import YOLO

# # model = YOLO('yolo11n-pose.pt')

# # results = model.track(source=0, show=True, tracker="botsort.yaml")



# # import cv2

# # from ultralytics import YOLO

# # # Load the YOLO model
# # model = YOLO("yolo11n.pt")

# # # Open the video file
# # # video_path = "path/to/your/video/file.mp4"
# # cap = cv2.VideoCapture(0)

# # # Loop through the video frames
# # while cap.isOpened():
# #     # Read a frame from the video
# #     success, frame = cap.read()

# #     if success:
# #         # Run YOLO inference on the frame
# #         results = model.track(frame, classes=[0], persist=True)
# #         print("=> Results: ", results)
        
# #         print('=> Boxes: ')
# #         for r in results:
# #             print(r.boxes)

# #         # Visualize the results on the frame
# #         annotated_frame = results[0].plot()
# #         print("=> Annotated Frame: ", annotated_frame)

# #         # Display the annotated frame
# #         cv2.imshow("YOLO Inference", annotated_frame)
# #         # break

# #         # Break the loop if 'q' is pressed
# #         if cv2.waitKey(1) & 0xFF == ord("q"):
# #             break
# #     else:
# #         # Break the loop if the end of the video is reached
# #         break

# # # Release the video capture object and close the display window
# # cap.release()
# # cv2.destroyAllWindows()





# # from collections import defaultdict

# # import cv2
# # import numpy as np

# # from ultralytics import YOLO

# # # Load the YOLO11 model
# # model = YOLO("yolo11n.pt")

# # # Open the video file
# # cap = cv2.VideoCapture(0)

# # # Store the track history
# # track_history = defaultdict(lambda: [])

# # # Loop through the video frames
# # while cap.isOpened():
# #     # Read a frame from the video
# #     success, frame = cap.read()

# #     if success:
# #         # Run YOLO11 tracking on the frame, persisting tracks between frames
# #         results = model.track(frame, persist=True, classes=[0])

# #         # Get the boxes and track IDs
# #         boxes = results[0].boxes.xywh.cpu()
# #         track_ids = results[0].boxes.id.int().cpu().tolist()

# #         # Visualize the results on the frame
# #         annotated_frame = results[0].plot()

# #         # Plot the tracks
# #         for box, track_id in zip(boxes, track_ids):
# #             x, y, w, h = box
# #             track = track_history[track_id]
# #             track.append((float(x), float(y)))  # x, y center point
# #             if len(track) > 30:  # retain 90 tracks for 90 frames
# #                 track.pop(0)

# #             # Draw the tracking lines
# #             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
# #             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

# #         # Display the annotated frame
# #         cv2.imshow("YOLO11 Tracking", annotated_frame)

# #         # Break the loop if 'q' is pressed
# #         if cv2.waitKey(1) & 0xFF == ord("q"):
# #             break
# #     else:
# #         # Break the loop if the end of the video is reached
# #         break

# # # Release the video capture object and close the display window
# # cap.release()
# # cv2.destroyAllWindows()








# # import cv2
# # from ultralytics import YOLO

# # # Load a YOLO model
# # model = YOLO("yolo11n.pt")  # Pretrained YOLO11n model

# # # Initialize the webcam (0 is the default camera index)
# # cap = cv2.VideoCapture(0)

# # # Loop through the webcam frames
# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if success:
# #         # Run YOLO inference on the webcam frame
# #         results = model.track(frame, stream=True, classes=[0])
# #         print(f"=> Results: {results}")

# #         for result in results:
# #             # print(f"=> Results: {result}")
# #             print(f"=> Results-Boxes: {result.boxes}")
# #             # Annotate the frame with detection results
# #             annotated_frame = result.plot()

# #             # Display the frame with results
# #             cv2.imshow("YOLO Webcam Inference", annotated_frame)
# #         # break

# #         # Break the loop if 'q' is pressed
# #         if cv2.waitKey(1) & 0xFF == ord("q"):
# #             break
# #     else:
# #         print("Failed to read from webcam.")
# #         break

# # # Release resources and close windows
# # cap.release()
# # cv2.destroyAllWindows()





# # import cv2
# # from ultralytics import YOLO

# # from deep_sort.utils.parser import get_config
# # from deep_sort.deep_sort import DeepSort
# # from deep_sort.sort.tracker import Tracker

# # import numpy as np

# # # Load a YOLO model
# # model = YOLO("yolo11n.pt")  # Pretrained YOLO11n model

# # deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
# # tracker = DeepSort(model_path=deep_sort_weights, max_age=70)


# # # Initialize the webcam (0 is the default camera index)
# # # cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture('/home/ultratechsystems/Desktop/AI_Projects/MOT/repos/norfair/video-samples/video.mp4')

# # # Loop through the webcam frames
# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if success:
# #         # Run YOLO tracking on the webcam frame
# #         results = model.track(frame, stream=True, classes=[0])  # Class 0 for "person"
# #         print(f"=> Results: {results}")

# #         for result in results:
# #             # Annotate the frame with detection results
# #             annotated_frame = result.plot()

# #             pred_cls = np.array(result.boxes.cls)
# #             conf = result.boxes.conf.detach().cpu().numpy()
# #             xyxy = result.boxes.xyxy.detach().cpu().numpy()
# #             bboxes_xywh = np.array(result.boxes.xywh.cpu().numpy(), dtype=float)

# #             tracks = tracker.update(bboxes_xywh, conf, frame)

# #             # Check if there are any detected boxes
# #             if result.boxes is not None:
# #                 for i, box in enumerate(result.boxes.data):
                    
# #                     # Ensure the box has enough values
# #                     print(f'=> Box: {box}')
# #                     print(f'=> Box: {result.boxes}')
# #                     print('-'*50)
# #                     if len(box) >= 7:
# #                         x1, y1, x2, y2, conf, cls, _ = box[:7]  # Extract bounding box and tracking ID
# #                         track_id = result.boxes.id[i]
# #                         # if int(track_id) > -1:  # Ensure the track ID is valid
# #                         cv2.putText(
# #                                 annotated_frame,
# #                                 f"ID: {int(track_id)}",
# #                                 (int(x1) + 10, int(y1) + 30),  # Position the text above the bounding box
# #                                 cv2.FONT_HERSHEY_SIMPLEX,
# #                                 0.5,  # Font scale
# #                                 (255, 0, 0),  # Font color (Blue)
# #                                 1,  # Line thickness
# #                                 cv2.LINE_AA,
# #                             )
# #                 print("="*70)

# #             # Display the frame with results
# #             cv2.imshow("YOLO Tracking Inference", annotated_frame)

# #         # Break the loop if 'q' is pressed
# #         if cv2.waitKey(1) & 0xFF == ord("q"):
# #             break
# #     else:
# #         print("Failed to read from webcam.")
# #         break

# # # Release resources and close windows
# # cap.release()
# # cv2.destroyAllWindows()



# # import cv2
# # from ultralytics import YOLO
# # import torch
# # import torchreid
# # from torchvision import transforms
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Initialize the YOLO model
# # model = YOLO("yolov8n.pt")  # Replace with your YOLO model

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',  # Lightweight ReID model
# #     num_classes=1000,
# #     pretrained=True
# # )
# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),  # Standard ReID input size
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Helper function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).numpy()

# # # Create a database of known embeddings
# # def create_known_database(image_paths, names, model, transform):
# #     database = []
# #     for image_path, name in zip(image_paths, names):
# #         image = cv2.imread(image_path)
# #         embedding = extract_embedding(image, model, transform)
# #         database.append({'name': name, 'embedding': embedding})
# #     return database

# # # Compare query embedding to the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for entry in database:
# #         similarity = cosine_similarity(
# #             query_embedding.reshape(1, -1),
# #             entry['embedding'].reshape(1, -1)
# #         )[0][0]
# #         if similarity > best_similarity and similarity > threshold:
# #             best_match = entry['name']
# #             best_similarity = similarity
# #     return best_match, best_similarity

# # # Initialize the webcam
# # cap = cv2.VideoCapture(0)

# # # Example: Known persons database (provide image paths and names)
# # known_persons = [
# #     {"path": "/home/ultratechsystems/Desktop/AI_Projects/MOT/repos/dummy-samples/Manish.jpeg", "name": "Manish"},
# #     # {"path": "known_person2.jpg", "name": "Person 2"},
# # ]
# # database = create_known_database(
# #     [person["path"] for person in known_persons],
# #     [person["name"] for person in known_persons],
# #     reid_model,
# #     transform
# # )

# # # Loop through the webcam frames
# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if success:
# #         # Run YOLO inference
# #         results = model.track(frame, stream=True, classes=[0])  # Person class only
        
# #         for result in results:
# #             for box in result.boxes.xyxy:  # Bounding boxes
# #                 x1, y1, x2, y2 = map(int, box)  # Get box coordinates
# #                 person_crop = frame[y1:y2, x1:x2]  # Crop the detected person

# #                 # Extract query embedding
# #                 query_embedding = extract_embedding(person_crop, reid_model, transform)

# #                 # Match with the known database
# #                 match, similarity = match_embedding(query_embedding, database)
# #                 label = match if match else "Unknown"

# #                 # Draw the bounding box and label
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                 cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# #         # Display the frame
# #         cv2.imshow("Webcam", frame)

# #         # Exit loop on 'q'
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #     else:
# #         print("Failed to read from webcam.")
# #         break

# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()









# # import cv2
# # from ultralytics import YOLO
# # import torch
# # import torchreid
# # from torchvision import transforms
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Initialize the YOLO model
# # model = YOLO("yolov8n.pt")  # Replace with your YOLO model

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',  # Lightweight ReID model
# #     num_classes=1000,
# #     pretrained=True
# # )
# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),  # Standard ReID input size
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Helper function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).numpy()

# # # Create a database of known embeddings
# # def create_known_database(image_paths, names, model, transform):
# #     database = []
# #     for image_path, name in zip(image_paths, names):
# #         image = cv2.imread(image_path)
# #         embedding = extract_embedding(image, model, transform)
# #         database.append({'name': name, 'embedding': embedding})
# #     return database

# # # Compare query embedding to the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for entry in database:
# #         similarity = cosine_similarity(
# #             query_embedding.reshape(1, -1),
# #             entry['embedding'].reshape(1, -1)
# #         )[0][0]
# #         if similarity > best_similarity and similarity > threshold:
# #             best_match = entry['name']
# #             best_similarity = similarity
# #     return best_match, best_similarity

# # # Initialize the webcam
# # cap = cv2.VideoCapture(0)

# # # Example: Known persons database (provide image paths and names)
# # known_persons = [
# #     {"path": "/home/ultratechsystems/Desktop/AI_Projects/MOT/repos/dummy-samples/Manish.jpeg", "name": "Manish"},
# #     # {"path": "known_person2.jpg", "name": "Person 2"},
# # ]
# # database = create_known_database(
# #     [person["path"] for person in known_persons],
# #     [person["name"] for person in known_persons],
# #     reid_model,
# #     transform
# # )

# # # Initialize a dictionary to store tracks
# # tracks = {}

# # # Loop through the webcam frames
# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if success:
# #         # Run YOLO inference
# #         results = model.track(frame, stream=True, classes=[0])  # Person class only
        
# #         for result in results:
# #             for box in result.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
# #                 track_id = int(box.id[0]) if box.id is not None else None

# #                 if track_id is not None:
# #                     # Get the center of the bounding box
# #                     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

# #                     # Update the track history
# #                     if track_id not in tracks:
# #                         tracks[track_id] = []
# #                     tracks[track_id].append((cx, cy))

# #                     # Limit the number of points to avoid excessive memory usage
# #                     if len(tracks[track_id]) > 50:
# #                         tracks[track_id] = tracks[track_id][-50:]

# #                     # Draw the track
# #                     for i in range(1, len(tracks[track_id])):
# #                         cv2.line(frame, tracks[track_id][i - 1], tracks[track_id][i], (0, 255, 0), 2)

# #                 # Crop the detected person for ReID
# #                 person_crop = frame[y1:y2, x1:x2]

# #                 # Extract query embedding
# #                 query_embedding = extract_embedding(person_crop, reid_model, transform)

# #                 # Match with the known database
# #                 match, similarity = match_embedding(query_embedding, database)
# #                 label = match if match else "Unknown"

# #                 # Draw the bounding box and label
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
# #                 cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

# #         # Display the frame
# #         cv2.imshow("Webcam", frame)

# #         # Exit loop on 'q'
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #     else:
# #         print("Failed to read from webcam.")
# #         break

# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()







# # import cv2
# # import torch
# # import torchreid
# # from torchvision import transforms
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # from tkinter import Tk, simpledialog

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',  # Lightweight ReID model
# #     num_classes=1000,
# #     pretrained=True
# # )
# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),  # Standard ReID input size
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Initialize the YOLO model
# # from ultralytics import YOLO
# # model = YOLO("yolo11n.pt")  # Replace with your YOLO model

# # # Database to store known embeddings
# # database = []

# # # Function to add a new person to the database
# # def add_person_to_database(name, person_crop, model, transform):
# #     embedding = extract_embedding(person_crop, model, transform)
# #     database.append({'name': name, 'embedding': embedding})
# #     print(f"Added new person: {name}")

# # # Function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).numpy()

# # # Function to match query embeddings with the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for entry in database:
# #         similarity = cosine_similarity(
# #             query_embedding.reshape(1, -1),
# #             entry['embedding'].reshape(1, -1)
# #         )[0][0]
# #         if similarity > best_similarity and similarity > threshold:
# #             best_match = entry['name']
# #             best_similarity = similarity
# #     return best_match, best_similarity

# # # Function to prompt for name using Tkinter
# # def get_person_name():
# #     root = Tk()
# #     root.withdraw()  # Hide the main Tkinter window
# #     name = simpledialog.askstring("Input", "Enter the name of the person:")
# #     root.destroy()  # Destroy the Tkinter window
# #     return name

# # # Initialize webcam
# # cap = cv2.VideoCapture(0)
# # adding_new_person = False
# # current_name = ""

# # print("Press 'n' to add a new person, 'q' to quit.")

# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if success:
# #         # Run YOLO inference
# #         results = model.track(frame, stream=True, classes=[0])  # Person class only

# #         for result in results:
# #             for box in result.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
# #                 person_crop = frame[y1:y2, x1:x2]  # Crop the detected person

# #                 # If adding a new person
# #                 if adding_new_person:
# #                     cv2.putText(frame, "Press 'c' to capture image", (50, 50),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

# #                     # Wait for the user to capture the image
# #                     if cv2.waitKey(1) & 0xFF == ord('c'):
# #                         add_person_to_database(current_name, person_crop, reid_model, transform)
# #                         adding_new_person = False

# #                 # If not adding, process for recognition
# #                 else:
# #                     query_embedding = extract_embedding(person_crop, reid_model, transform)
# #                     match, similarity = match_embedding(query_embedding, database)
# #                     label = match if match else "Unknown"

# #                     # Draw bounding box and label
# #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

# #         # Display the frame
# #         cv2.imshow("Webcam", frame)

# #         # Key bindings
# #         key = cv2.waitKey(1) & 0xFF

# #         if key == ord('n'):  # Start adding a new person
# #             adding_new_person = True
# #             current_name = get_person_name()  # Get the name using the popup
# #             if not current_name:  # If the user cancels, stop adding
# #                 adding_new_person = False
# #             else:
# #                 print("Move the person into the camera view.")

# #         if key == ord('q'):  # Quit
# #             break
# #     else:
# #         print("Failed to read from webcam.")
# #         break

# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()










# # import cv2
# # import torch
# # import torchreid
# # from torchvision import transforms
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # from tkinter import Tk, simpledialog

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',  # Lightweight ReID model
# #     num_classes=1000,
# #     pretrained=True
# # )
# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),  # Standard ReID input size
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Initialize the YOLO model
# # from ultralytics import YOLO
# # model = YOLO("yolov8n.pt")  # Replace with your YOLO model

# # # Database to store known embeddings
# # database = {}

# # # Function to add a new person to the database
# # def add_person_to_database(name, images, model, transform):
# #     embeddings = [extract_embedding(image, model, transform) for image in images]
# #     database[name] = embeddings
# #     print(f"Added new person: {name} with {len(images)} images.")

# # # Function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).numpy()

# # # Function to match query embeddings with the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for name, embeddings in database.items():
# #         for embedding in embeddings:
# #             similarity = cosine_similarity(
# #                 query_embedding.reshape(1, -1),
# #                 embedding.reshape(1, -1)
# #             )[0][0]
# #             if similarity > best_similarity and similarity > threshold:
# #                 best_match = name
# #                 best_similarity = similarity
# #     return best_match, best_similarity

# # # Function to prompt for name using simpledialog
# # def get_person_name():
# #     root = Tk()
# #     root.withdraw()  # Hide the main Tkinter root window
# #     name = simpledialog.askstring("Input", "Enter the name of the person:")
# #     root.destroy()  # Destroy the Tkinter root window
# #     return name

# # # Initialize webcam
# # cap = cv2.VideoCapture(0)
# # adding_new_person = False
# # current_name = ""
# # capture_images = []
# # view_directions = ["Front", "Back", "Right", "Left"]
# # current_view = 0

# # print("Press 'n' to add a new person, 'q' to quit.")

# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if success:
# #         # Run YOLO inference
# #         results = model.track(frame, stream=True, classes=[0], persist=True)  # Person class only

# #         for result in results:
# #             for box in result.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
# #                 person_crop = frame[y1:y2, x1:x2]  # Crop the detected person

# #                 if adding_new_person:
# #                     # Display instructions for capturing images
# #                     cv2.putText(frame, f"Capture {view_directions[current_view]} View", (50, 50),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

# #                     # Wait for the user to capture the image
# #                     if cv2.waitKey(1) & 0xFF == ord('c'):
# #                         capture_images.append(person_crop)
# #                         print(f"Captured {view_directions[current_view]} view.")
# #                         current_view += 1

# #                         # If all views are captured, add to the database
# #                         if current_view == len(view_directions):
# #                             add_person_to_database(current_name, capture_images, reid_model, transform)
# #                             adding_new_person = False
# #                             capture_images = []
# #                             current_view = 0

# #                 else:
# #                     # If not adding, process for recognition
# #                     query_embedding = extract_embedding(person_crop, reid_model, transform)
# #                     match, similarity = match_embedding(query_embedding, database)
# #                     label = match if match else "Unknown"

# #                     # Draw bounding box and label
# #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

# #         # Display the frame
# #         cv2.imshow("Webcam", frame)

# #         # Key bindings
# #         key = cv2.waitKey(1) & 0xFF

# #         if key == ord('n'):  # Start adding a new person
# #             adding_new_person = True
# #             current_name = get_person_name()  # Get the name using the popup
# #             if not current_name:
# #                 adding_new_person = False
# #             else:
# #                 print("Move the person into the camera view and capture images.")

# #         if key == ord('q'):  # Quit
# #             break
# #     else:
# #         print("Failed to read from webcam.")
# #         break

# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()



# #------------------------------------------------------------------




# # import cv2
# # import os
# # import time
# # import torch
# # import torchreid
# # from torchvision import transforms
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # from tkinter import Tk, simpledialog

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',  # Lightweight ReID model
# #     num_classes=1000,
# #     pretrained=True
# # )

# # # reid_model = reid_model.cuda()

# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),  # Standard ReID input size
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Initialize the YOLO model
# # from ultralytics import YOLO
# # model = YOLO("yolo11n.pt")  # Replace with your YOLO model

# # # Database directory
# # database_dir = './database'

# # # Function to add a new person to the database
# # def add_person_to_database(name, images, model, transform):
# #     person_dir = os.path.join(database_dir, name)
# #     if not os.path.exists(person_dir):
# #         os.makedirs(person_dir)
    
# #     # Save images
# #     for idx, image in enumerate(images):
# #         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
# #         cv2.imwrite(image_filename, image)
# #         print(f"Saved image to {image_filename}")
    
# #     embeddings = [extract_embedding(image, model, transform) for image in images]
# #     return embeddings

# # # Function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).numpy()

# # # Function to match query embeddings with the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for name, embeddings in database.items():
# #         for embedding in embeddings:
# #             similarity = cosine_similarity(
# #                 query_embedding.reshape(1, -1),
# #                 embedding.reshape(1, -1)
# #             )[0][0]
# #             if similarity > best_similarity and similarity > threshold:
# #                 best_match = name
# #                 best_similarity = similarity
# #     return best_match, best_similarity

# # # Function to load all images and embeddings from the database
# # def load_database(database_dir, model, transform):
# #     database = {}
# #     if os.path.exists(database_dir):
# #         for person_name in os.listdir(database_dir):
# #             person_dir = os.path.join(database_dir, person_name)
# #             if os.path.isdir(person_dir):
# #                 embeddings = []
# #                 for image_filename in os.listdir(person_dir):
# #                     image_path = os.path.join(person_dir, image_filename)
# #                     image = cv2.imread(image_path)
# #                     if image is not None:
# #                         embeddings.append(extract_embedding(image, model, transform))
# #                 database[person_name] = embeddings
# #     return database

# # # Function to prompt for name using simpledialog
# # def get_person_name():
# #     root = Tk()
# #     root.withdraw()  # Hide the main Tkinter root window
# #     name = simpledialog.askstring("Input", "Enter the name of the person:")
# #     root.destroy()  # Destroy the Tkinter root window
# #     return name

# # # Initialize webcam
# # ip_camera_url = "rtsp://foscam@123:foscam@123@192.168.1.104:88/videoMain"
# # cap = cv2.VideoCapture(ip_camera_url)
# # # cap = cv2.VideoCapture(0)
# # adding_new_person = False
# # current_name = ""
# # capture_images = []
# # view_directions = ["Front", "Back", "Right", "Left"]
# # current_view = 0

# # # Load existing database
# # database = load_database(database_dir, reid_model, transform)

# # print("Press 'n' to add a new person, 'q' to quit.")

# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if success:
# #         # Run YOLO inference
# #         results = model.track(frame, stream=True, classes=[0], persist=True)  # Person class only

# #         for result in results:
# #             for box in result.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
# #                 person_crop = frame[y1:y2, x1:x2]  # Crop the detected person

# #                 if adding_new_person:
# #                     # Display instructions for capturing images
# #                     cv2.putText(frame, f"Capture {view_directions[current_view]} View", (50, 50),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

# #                     # Wait for the user to capture the image
# #                     if cv2.waitKey(1) & 0xFF == ord('c'):
# #                         capture_images.append(person_crop)
# #                         print(f"Captured {view_directions[current_view]} view.")
# #                         current_view += 1

# #                         # If all views are captured, add to the database
# #                         if current_view == len(view_directions):
# #                             new_embeddings = add_person_to_database(current_name, capture_images, reid_model, transform)
# #                             if current_name not in database:
# #                                 database[current_name] = new_embeddings
# #                             adding_new_person = False
# #                             capture_images = []
# #                             current_view = 0

# #                 else:
# #                     # If not adding, process for recognition
# #                     query_embedding = extract_embedding(person_crop, reid_model, transform)
# #                     match, similarity = match_embedding(query_embedding, database)
# #                     label = match if match else "Unknown"

# #                     # Draw bounding box and label
# #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

# #         # Display the frame
# #         cv2.imshow("Webcam", frame)

# #         # Key bindings
# #         key = cv2.waitKey(1) & 0xFF

# #         if key == ord('n'):  # Start adding a new person
# #             adding_new_person = True
# #             current_name = get_person_name()  # Get the name using the popup
# #             if not current_name:
# #                 adding_new_person = False
# #             else:
# #                 print("Move the person into the camera view and capture images.")

# #         if key == ord('q'):  # Quit
# #             break
# #     else:
# #         print("Failed to read from webcam.")
# #         break

# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()










# # import cv2
# # import os
# # import time
# # import torch
# # import torchreid
# # from torchvision import transforms
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # from tkinter import Tk, simpledialog

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',
# #     num_classes=1000,
# #     pretrained=True
# # )

# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Initialize the YOLO model
# # from ultralytics import YOLO
# # model = YOLO("yolo11n.pt")

# # # Database directory
# # database_dir = './database'

# # # Function to add a new person to the database
# # def add_person_to_database(name, images, model, transform):
# #     person_dir = os.path.join(database_dir, name)
# #     if not os.path.exists(person_dir):
# #         os.makedirs(person_dir)
# #     for idx, image in enumerate(images):
# #         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
# #         cv2.imwrite(image_filename, image)
# #     embeddings = [extract_embedding(image, model, transform) for image in images]
# #     return embeddings

# # # Function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0)
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).numpy()

# # # Function to match query embeddings with the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for name, embeddings in database.items():
# #         for embedding in embeddings:
# #             similarity = cosine_similarity(
# #                 query_embedding.reshape(1, -1),
# #                 embedding.reshape(1, -1)
# #             )[0][0]
# #             if similarity > best_similarity and similarity > threshold:
# #                 best_match = name
# #                 best_similarity = similarity
# #     return best_match, best_similarity

# # # Function to load all images and embeddings from the database
# # def load_database(database_dir, model, transform):
# #     database = {}
# #     if os.path.exists(database_dir):
# #         for person_name in os.listdir(database_dir):
# #             person_dir = os.path.join(database_dir, person_name)
# #             if os.path.isdir(person_dir):
# #                 embeddings = []
# #                 for image_filename in os.listdir(person_dir):
# #                     image_path = os.path.join(person_dir, image_filename)
# #                     image = cv2.imread(image_path)
# #                     if image is not None:
# #                         embeddings.append(extract_embedding(image, model, transform))
# #                 database[person_name] = embeddings
# #     return database

# # # Function to prompt for name using simpledialog
# # def get_person_name():
# #     root = Tk()
# #     root.withdraw()
# #     name = simpledialog.askstring("Input", "Enter the name of the person:")
# #     root.destroy()
# #     return name

# # # Initialize webcam
# # ip_camera_url = "rtsp://admin:Lucky786$1@192.168.1.108:554/Streaming/Channels/101"
# # # ip_camera_url = "rtsp://foscam@123:foscam@123@192.168.1.104:88/videoMain"

# # cap = cv2.VideoCapture(ip_camera_url)

# # cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size to 3 frames

# # # Globals for adding a person
# # adding_new_person = False
# # current_name = ""
# # capture_images = []
# # view_directions = ["Front", "Back", "Right", "Left"]
# # current_view = 0

# # # Load existing database
# # database = load_database(database_dir, reid_model, transform)

# # print("Press 'n' to add a new person, 'q' to quit.")

# # frame_count = 0

# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if not success:
# #         print("Failed to read from camera.")
# #         break

# #     frame_count += 1
# #     if frame_count % 2 != 0:
# #             continue

# #     # Run YOLO inference
# #     results = model.track(frame, stream=True, classes=[0], persist=True)

# #     for result in results:
# #         for box in result.boxes:
# #             x1, y1, x2, y2 = map(int, box.xyxy[0])
# #             person_crop = frame[y1:y2, x1:x2]

# #             if adding_new_person:
# #                 cv2.putText(frame, f"Capture {view_directions[current_view]} View", (50, 50),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

# #                 if cv2.waitKey(1) & 0xFF == ord('c'):
# #                     capture_images.append(person_crop)
# #                     print(f"Captured {view_directions[current_view]} view.")
# #                     current_view += 1

# #                     if current_view == len(view_directions):
# #                         new_embeddings = add_person_to_database(current_name, capture_images, reid_model, transform)
# #                         database[current_name] = new_embeddings
# #                         adding_new_person = False
# #                         capture_images = []
# #                         current_view = 0
# #             else:
# #                 query_embedding = extract_embedding(person_crop, reid_model, transform)
# #                 match, similarity = match_embedding(query_embedding, database)
# #                 label = match if match else "Unknown"
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
# #                 cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 250, 0), 2)

# #     frame_dis = cv2.resize(frame, (960, 540))
# #     cv2.imshow("Webcam", frame_dis)

# #     key = cv2.waitKey(1) & 0xFF
# #     if key == ord('n'):
# #         adding_new_person = True
# #         current_name = get_person_name()
# #         if not current_name:
# #             adding_new_person = False
# #         else:
# #             print("Move the person into the camera view and capture images.")

# #     if key == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()







# # import cv2
# # import os
# # import time
# # import torch
# # import torchreid
# # from torchvision import transforms
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # from threading import Thread

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',
# #     num_classes=1000,
# #     pretrained=True
# # )

# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Initialize the YOLO model
# # from ultralytics import YOLO
# # model = YOLO("yolo11n.pt")

# # # Database directory
# # database_dir = './database'

# # # Function to add a new person to the database
# # def add_person_to_database(name, images, model, transform):
# #     person_dir = os.path.join(database_dir, name)
# #     if not os.path.exists(person_dir):
# #         os.makedirs(person_dir)
# #     for idx, image in enumerate(images):
# #         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
# #         cv2.imwrite(image_filename, image)
# #     embeddings = [extract_embedding(image, model, transform) for image in images]
# #     return embeddings

# # # Function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0)
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).numpy()

# # # Function to match query embeddings with the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for name, embeddings in database.items():
# #         for embedding in embeddings:
# #             similarity = cosine_similarity(
# #                 query_embedding.reshape(1, -1),
# #                 embedding.reshape(1, -1)
# #             )[0][0]
# #             if similarity > best_similarity and similarity > threshold:
# #                 best_match = name
# #                 best_similarity = similarity
# #     return best_match, best_similarity

# # # Function to load all images and embeddings from the database
# # def load_database(database_dir, model, transform):
# #     database = {}
# #     if os.path.exists(database_dir):
# #         for person_name in os.listdir(database_dir):
# #             person_dir = os.path.join(database_dir, person_name)
# #             if os.path.isdir(person_dir):
# #                 embeddings = []
# #                 for image_filename in os.listdir(person_dir):
# #                     image_path = os.path.join(person_dir, image_filename)
# #                     image = cv2.imread(image_path)
# #                     if image is not None:
# #                         embeddings.append(extract_embedding(image, model, transform))
# #                 database[person_name] = embeddings
# #     return database

# # # Initialize webcam URLs
# # camera_urls = [
# #     "rtsp://admin:Lucky786$1@192.168.1.110:554/Streaming/Channels/101",
# #     "rtsp://admin:Lucky786$1@192.168.1.110:554/Streaming/Channels/201"
# # ]

# # # Globals for database
# # database = load_database(database_dir, reid_model, transform)

# # # Function to process a single camera feed
# # def process_camera_feed(camera_url, camera_id):
# #     print(f"Initializing camera {camera_id} with URL: {camera_url}")
# #     cap = cv2.VideoCapture(camera_url)
# #     if not cap.isOpened():
# #         print(f"Error: Camera {camera_id} failed to open. Check the RTSP URL or connection.")
# #         return

# #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size to 3 frames
# #     frame_count = 0

# #     while cap.isOpened():
# #         success, frame = cap.read()
# #         if not success:
# #             print(f"Error: Camera {camera_id} failed to read frame.")
# #             break

# #         frame_count += 1
# #         if frame_count % 2 != 0:
# #             continue

# #         # Run YOLO inference
# #         results = model.track(frame, stream=True, classes=[0], persist=True)

# #         for result in results:
# #             for box in result.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
# #                 person_crop = frame[y1:y2, x1:x2]

# #                 query_embedding = extract_embedding(person_crop, reid_model, transform)
# #                 match, similarity = match_embedding(query_embedding, database)
# #                 label = match if match else "Unknown"
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
# #                 cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)

# #         frame_dis = cv2.resize(frame, (960, 540))
# #         cv2.imshow(f"Camera {camera_id}", frame_dis)

# #         key = cv2.waitKey(1) & 0xFF
# #         if key == ord('q'):
# #             print(f"Exiting camera {camera_id}...")
# #             cap.release()
# #             cv2.destroyWindow(f"Camera {camera_id}")
# #             return

# # # Start threads for each camera
# # threads = []
# # def start_camera_threads():
# #     for i, url in enumerate(camera_urls):
# #         print(f"Starting thread for Camera {i}")
# #         t = Thread(target=process_camera_feed, args=(url, i))
# #         t.daemon = True  # Ensure threads exit when the main program exits
# #         t.start()
# #         threads.append(t)

# # start_camera_threads()

# # # Wait for 'q' to quit
# # print("Press 'q' to quit all camera feeds.")
# # while True:
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cv2.destroyAllWindows()








# # import cv2
# # import os
# # import time
# # import torch
# # import torchreid
# # from torchvision import transforms
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # from threading import Thread, Lock
# # from queue import Queue

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # print(f"=>> Device Used for Inference: {device}")

# # # Initialize the ReID model
# # reid_model = torchreid.models.build_model(
# #     name='osnet_x1_0',
# #     num_classes=1000,
# #     pretrained=True
# # ).to(device)

# # reid_model.eval()

# # # Preprocessing function for the ReID model
# # transform = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.Resize((256, 128)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Initialize the YOLO model
# # from ultralytics import YOLO
# # model = YOLO("yolo11n.pt").to(device)
# # # model.fuse()  # Fuse model layers for faster inference
# # # if torch.cuda.is_available():
# # #     model.half()  # Use FP16 for faster inference on GPU

# # # Database directory
# # database_dir = './database'

# # # Function to add a new person to the database
# # def add_person_to_database(name, images, model, transform):
# #     person_dir = os.path.join(database_dir, name)
# #     if not os.path.exists(person_dir):
# #         os.makedirs(person_dir)
# #     for idx, image in enumerate(images):
# #         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
# #         cv2.imwrite(image_filename, image)
# #     embeddings = [extract_embedding(image, model, transform) for image in images]
# #     return embeddings

# # # Function to extract embeddings
# # def extract_embedding(image, model, transform):
# #     image_tensor = transform(image).unsqueeze(0).to(device)
# #     with torch.no_grad():
# #         embedding = model(image_tensor)
# #     return embedding.squeeze(0).cpu().numpy()

# # # Function to match query embeddings with the database
# # def match_embedding(query_embedding, database, threshold=0.7):
# #     best_match = None
# #     best_similarity = 0
# #     for name, embeddings in database.items():
# #         for embedding in embeddings:
# #             similarity = cosine_similarity(
# #                 query_embedding.reshape(1, -1),
# #                 embedding.reshape(1, -1)
# #             )[0][0]
# #             if similarity > best_similarity and similarity > threshold:
# #                 best_match = name
# #                 best_similarity = similarity
# #     return best_match, best_similarity

# # # Function to load all images and embeddings from the database
# # def load_database(database_dir, model, transform):
# #     database = {}
# #     if os.path.exists(database_dir):
# #         for person_name in os.listdir(database_dir):
# #             person_dir = os.path.join(database_dir, person_name)
# #             if os.path.isdir(person_dir):
# #                 embeddings = []
# #                 for image_filename in os.listdir(person_dir):
# #                     image_path = os.path.join(person_dir, image_filename)
# #                     image = cv2.imread(image_path)
# #                     if image is not None:
# #                         embeddings.append(extract_embedding(image, model, transform))
# #                 database[person_name] = embeddings
# #     return database

# # # Initialize webcam URLs
# # camera_urls = [
# #     "rtsp://admin:Lucky786$1@192.168.1.112:554/Streaming/Channels/101",
# #     "rtsp://admin:Lucky786$1@192.168.1.112:554/Streaming/Channels/201"
# # ]

# # # Globals for database
# # database = load_database(database_dir, reid_model, transform)

# # # Queues for camera frames
# # frame_queues = [Queue(maxsize=3) for _ in camera_urls]
# # lock = Lock()

# # # Function to process a single camera feed
# # def process_camera_feed(camera_url, camera_id):
# #     print(f"Initializing camera {camera_id} with URL: {camera_url}")
# #     cap = cv2.VideoCapture(camera_url)
# #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# #     if not cap.isOpened():
# #         print(f"Error: Camera {camera_id} failed to open. Check the RTSP URL or connection.")
# #         return

# #     frame_count = 0
# #     while cap.isOpened():
# #         success, frame = cap.read()
# #         if not success:
# #             print(f"Error: Camera {camera_id} failed to read frame.")
# #             break

# #         frame_count+=1
# #         if frame_count % 2 != 0:
# #             continue

# #         # Run YOLO inference
# #         results = model.track(frame, stream=True, classes=[0], persist=True)

# #         for result in results:
# #             for box in result.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
# #                 person_crop = frame[y1:y2, x1:x2]

# #                 query_embedding = extract_embedding(person_crop, reid_model, transform)
# #                 match, similarity = match_embedding(query_embedding, database)
# #                 label = match if match else "Unknown"
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
# #                 cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)

# #         # Resize and push the frame to the queue
# #         frame_resized = cv2.resize(frame, (640, 360))
# #         with lock:
# #             if not frame_queues[camera_id].full():
# #                 frame_queues[camera_id].put(frame_resized)

# #     cap.release()

# # # Start threads for each camera
# # threads = []
# # for i, url in enumerate(camera_urls):
# #     t = Thread(target=process_camera_feed, args=(url, i))
# #     t.daemon = True
# #     t.start()
# #     threads.append(t)

# # # Display frames in the main thread
# # while True:
# #     for i in range(len(camera_urls)):
# #         with lock:
# #             if not frame_queues[i].empty():
# #                 frame = frame_queues[i].get()
# #                 cv2.imshow(f"Camera {i}", frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cv2.destroyAllWindows()













# import cv2
# import os
# import time
# import torch
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from threading import Thread, Lock
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog


# # if torch.cuda.is_available():
# #     device = torch.device("cuda")
# #     print("GPU:", torch.cuda.get_device_name(0))
# #     print("Supports FP16:", torch.cuda.get_device_capability(0))


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize the ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)

# reid_model.eval()

# # Preprocessing function for the ReID model
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize the YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# # Database directory
# database_dir = './database'

# # Function to add a new person to the database
# def add_person_to_database(name, images, model, transform):
#     person_dir = os.path.join(database_dir, name)
#     if not os.path.exists(person_dir):
#         os.makedirs(person_dir)
#     for idx, image in enumerate(images):
#         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
#         cv2.imwrite(image_filename, image)
#     embeddings = [extract_embedding(image, model, transform) for image in images]
#     return embeddings

# # Function to extract embeddings
# def extract_embedding(image, model, transform):
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

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

# # Function to load all images and embeddings from the database
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

# # Function to prompt for name using simpledialog
# def get_person_name():
#     root = Tk()
#     root.withdraw()  # Hide the main Tkinter root window
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()  # Destroy the Tkinter root window
#     return name

# # Initialize webcam URLs
# camera_urls = [
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
#     # "rtsp://foscam@123:foscam@123@192.168.1.227:88/videoMain", 
# ]

# # Globals for database
# database = load_database(database_dir, reid_model, transform)

# # Queues for camera frames
# frame_queues = [Queue(maxsize=4) for _ in camera_urls]
# lock = Lock()

# # Adding new person variables
# adding_new_person = False
# current_name = ""
# capture_images = []
# view_directions = ["Front", "Back", "Right", "Left"]
# current_view = 0

# # Function to process a single camera feed
# def process_camera_feed(camera_url, camera_id):
#     global adding_new_person, current_name, capture_images, current_view
#     print(f"Initializing camera {camera_id} with URL: {camera_url}")
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open. Check the RTSP URL or connection.")
#         return

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break

#         # Run YOLO inference
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])

#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 person_crop = frame[y1:y2, x1:x2]

#                 if adding_new_person:
#                     cv2.putText(frame, f"Capture {view_directions[current_view]} View", (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if cv2.waitKey(1) & 0xFF == ord('c'):
#                         capture_images.append(person_crop)
#                         print(f"Captured {view_directions[current_view]} view.")
#                         current_view += 1

#                         if current_view == len(view_directions):
#                             new_embeddings = add_person_to_database(current_name, capture_images, reid_model, transform)
#                             database[current_name] = new_embeddings
#                             adding_new_person = False
#                             capture_images = []
#                             current_view = 0
#                 else:
#                     # query_embedding = extract_embedding(person_crop, reid_model, transform)
#                     # match, similarity = match_embedding(query_embedding, database)
#                     # label = match if match else "Unknown"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     # cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
#                     #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)

#         # Resize and push the frame to the queue
#         frame_resized = cv2.resize(frame, (640, 480))
#         with lock:
#             if not frame_queues[camera_id].full():
#                 frame_queues[camera_id].put(frame_resized)

#     cap.release()

# # Start threads for each camera
# threads = []
# for i, url in enumerate(camera_urls):
#     t = Thread(target=process_camera_feed, args=(url, i))
#     t.daemon = True
#     t.start()
#     threads.append(t)

# # Display frames in the main thread
# while True:
#     for i in range(len(camera_urls)):
#         with lock:
#             if not frame_queues[i].empty():
#                 frame = frame_queues[i].get()
#                 cv2.imshow(f"Camera {i}", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('n') and not adding_new_person:
#         adding_new_person = True
#         current_name = get_person_name()
#         if not current_name:
#             adding_new_person = False

# cv2.destroyAllWindows()






#-------------------------------------------------------------------------------------------------------------






# import cv2
# import os
# import time
# import torch
# import torch.multiprocessing as mp
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# # Set multiprocessing start method to "spawn"
# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize the ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function for the ReID model
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize the YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# # Database directory
# database_dir = './database'

# # Function to add a new person to the database
# def add_person_to_database(name, images, model, transform):
#     person_dir = os.path.join(database_dir, name)
#     os.makedirs(person_dir, exist_ok=True)
    
#     embeddings = []
#     for idx, image in enumerate(images):
#         image_filename = os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg")
#         cv2.imwrite(image_filename, image)
#         embeddings.append(extract_embedding(image, model, transform))
    
#     return embeddings

# # Function to extract embeddings
# def extract_embedding(image, model, transform):
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

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

# # Function to load database
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

# # Function to get person name via Tkinter
# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# # Initialize webcam URLs
# camera_urls = [
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.119:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.135:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
# ]

# # Load database
# database = load_database(database_dir, reid_model, transform)

# # Adding new person variables
# adding_new_person = False
# current_name = ""
# capture_images = []
# view_directions = ["Front", "Back", "Right", "Left"]
# current_view = 0

# # Function to process a single camera feed using multiprocessing
# def process_camera_feed(camera_url, camera_id, frame_queue):
#     global adding_new_person, current_name, capture_images, current_view

#     print(f"Initializing camera {camera_id} with URL: {camera_url}")
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open. Check RTSP URL.")
#         return

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break

#         # Run YOLO inference
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])

#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 person_crop = frame[y1:y2, x1:x2]

#                 if adding_new_person:
#                     cv2.putText(frame, f"Capture {view_directions[current_view]} View", (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if cv2.waitKey(1) & 0xFF == ord('c'):
#                         capture_images.append(person_crop)
#                         print(f"Captured {view_directions[current_view]} view.")
#                         current_view += 1

#                         if current_view == len(view_directions):
#                             new_embeddings = add_person_to_database(current_name, capture_images, reid_model, transform)
#                             database[current_name] = new_embeddings
#                             adding_new_person = False
#                             capture_images = []
#                             current_view = 0
#                 else:
#                     query_embedding = extract_embedding(person_crop, reid_model, transform)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)

#         # Resize and send frame to queue
#         frame_resized = cv2.resize(frame, (854, 480))
#         if not frame_queue.full():
#             frame_queue.put(frame_resized)

#     cap.release()

# if __name__ == "__main__":
#     # Create multiprocessing queues
#     frame_queues = [mp.Queue(maxsize=4) for _ in camera_urls]

#     # Start camera processes
#     processes = []
#     for i, url in enumerate(camera_urls):
#         p = mp.Process(target=process_camera_feed, args=(url, i, frame_queues[i]))
#         p.daemon = True
#         p.start()
#         processes.append(p)

#     # Main display loop
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 frame = frame_queues[i].get()
#                 cv2.imshow(f"Camera {i}", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('n') and not adding_new_person:
#             adding_new_person = True
#             current_name = get_person_name()
#             if not current_name:
#                 adding_new_person = False

#     # Cleanup
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()









# import cv2
# import os
# import time
# import torch
# import torch.multiprocessing as mp
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# database_dir = './database'

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
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = reid_model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match, best_similarity = None, 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(
#                 query_embedding.reshape(1, -1),
#                 embedding.reshape(1, -1)
#             )[0][0]
#             if similarity > best_similarity and similarity > threshold:
#                 best_match, best_similarity = name, similarity
#     return best_match, best_similarity

# def load_database():
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image = cv2.imread(os.path.join(person_dir, image_filename))
#                     if image is not None:
#                         embeddings.append(extract_embedding(image))
#                 database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# camera_urls = [
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.119:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.135:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
# ]

# database = load_database()

# def process_camera_feed(camera_url, camera_id, frame_queue):
#     global adding_new_person, current_name, capture_images, current_view
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open.")
#         return

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break
        
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 person_crop = frame[y1:y2, x1:x2]
#                 if adding_new_person and camera_id == 0:
#                     cv2.putText(frame, f"Capture {view_directions[current_view]} View", (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if cv2.waitKey(1) & 0xFF == ord('c'):
#                         capture_images.append(person_crop)
#                         print(f"Captured {view_directions[current_view]} view.")
#                         current_view += 1
#                         if current_view == len(view_directions):
#                             database[current_name] = add_person_to_database(current_name, capture_images)
#                             adding_new_person, capture_images, current_view = False, [], 0
#                 else:
#                     query_embedding = extract_embedding(person_crop)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)
#         frame_queue.put(cv2.resize(frame, (854, 480)))
#     cap.release()

# if __name__ == "__main__":
#     frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
#     processes = [mp.Process(target=process_camera_feed, args=(url, i, frame_queues[i]), daemon=True) for i, url in enumerate(camera_urls)]
#     for p in processes:
#         p.start()
    
#     adding_new_person, capture_images, current_view = False, [], 0
#     view_directions = ["Front", "Back", "Right", "Left"]
    
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 cv2.imshow(f"Camera {i}", frame_queues[i].get())
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('n') and not adding_new_person:
#             adding_new_person, current_name = True, get_person_name()
#             if not current_name:
#                 adding_new_person = False
    
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()




















# import cv2
# import os
# import time
# import torch
# import torch.multiprocessing as mp
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# database_dir = './database'

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
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = reid_model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match, best_similarity = None, 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(
#                 query_embedding.reshape(1, -1),
#                 embedding.reshape(1, -1)
#             )[0][0]
#             if similarity > best_similarity and similarity > threshold:
#                 best_match, best_similarity = name, similarity
#     return best_match, best_similarity

# def load_database():
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image = cv2.imread(os.path.join(person_dir, image_filename))
#                     if image is not None:
#                         embeddings.append(extract_embedding(image))
#                 database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# camera_urls = [
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201"
# ]

# database = load_database()

# def process_camera_feed(camera_url, camera_id, frame_queue, shared_data):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open.")
#         return

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break
        
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 person_crop = frame[y1:y2, x1:x2]
#                 if shared_data['adding_new_person'].value and camera_id == 0:
#                     cv2.putText(frame, f"Capture {shared_data['view_directions'][shared_data['current_view'].value]} View", (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if cv2.waitKey(1) & 0xFF == ord('c'):
#                         shared_data['capture_images'].append(person_crop)
#                         print(f"Captured {shared_data['view_directions'][shared_data['current_view'].value]} view.")
#                         shared_data['current_view'].value += 1
#                         if shared_data['current_view'].value == len(shared_data['view_directions']):
#                             database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                             shared_data['adding_new_person'].value = False
#                             shared_data['capture_images'][:] = []
#                             shared_data['current_view'].value = 0
#                 else:
#                     query_embedding = extract_embedding(person_crop)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)
#         frame_queue.put(cv2.resize(frame, (854, 480)))
#     cap.release()

# if __name__ == "__main__":
#     manager = mp.Manager()
#     shared_data = manager.dict({
#         'adding_new_person': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"]
#     })

#     frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
#     processes = [mp.Process(target=process_camera_feed, args=(url, i, frame_queues[i], shared_data), daemon=True) for i, url in enumerate(camera_urls)]
#     for p in processes:
#         p.start()
    
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 cv2.imshow(f"Camera {i}", frame_queues[i].get())
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('n') and not shared_data['adding_new_person'].value:
#             shared_data['adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['adding_new_person'].value = False
    
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()
















# import cv2
# import os
# import time
# import torch
# import torch.multiprocessing as mp
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# database_dir = './database'

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
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = reid_model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match, best_similarity = None, 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(
#                 query_embedding.reshape(1, -1),
#                 embedding.reshape(1, -1)
#             )[0][0]
#             if similarity > best_similarity and similarity > threshold:
#                 best_match, best_similarity = name, similarity
#     return best_match, best_similarity

# def load_database():
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image = cv2.imread(os.path.join(person_dir, image_filename))
#                     if image is not None:
#                         embeddings.append(extract_embedding(image))
#                 database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# camera_urls = [
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.119:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.135:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
# ]

# database = load_database()

# def process_camera_feed(camera_url, camera_id, frame_queue, shared_data):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open.")
#         return

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break
        
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 person_crop = frame[y1:y2, x1:x2]
#                 if shared_data['adding_new_person'].value and camera_id == 0:
#                     cv2.putText(frame, f"Capture {shared_data['view_directions'][shared_data['current_view'].value]} View", (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if shared_data['capture_signal'].value:
#                         shared_data['capture_images'].append(person_crop)
#                         print(f"Captured {shared_data['view_directions'][shared_data['current_view'].value]} view.")
#                         shared_data['current_view'].value += 1
#                         shared_data['capture_signal'].value = False
#                         if shared_data['current_view'].value == len(shared_data['view_directions']):
#                             database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                             shared_data['adding_new_person'].value = False
#                             shared_data['capture_images'][:] = []
#                             shared_data['current_view'].value = 0
#                 else:
#                     query_embedding = extract_embedding(person_crop)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)
#         frame_queue.put(cv2.resize(frame, (854, 480)))
#     cap.release()

# if __name__ == "__main__":
#     manager = mp.Manager()
#     shared_data = manager.dict({
#         'adding_new_person': manager.Value('b', False),
#         'capture_signal': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"]
#     })

#     frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
#     processes = [mp.Process(target=process_camera_feed, args=(url, i, frame_queues[i], shared_data), daemon=True) for i, url in enumerate(camera_urls)]
#     for p in processes:
#         p.start()
    
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 cv2.imshow(f"Camera {i}", frame_queues[i].get())
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('n') and not shared_data['adding_new_person'].value:
#             shared_data['adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['adding_new_person'].value = False
#         elif key == ord('c'):
#             shared_data['capture_signal'].value = True
    
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()












#------------------------------------------------------------------------------------------




# import cv2
# import os
# import time
# import torch
# import torch.multiprocessing as mp
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# import logging
# logging.getLogger("ultralytics").setLevel(logging.WARNING)

# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# database_dir = './database'

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
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = reid_model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match, best_similarity = None, 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(
#                 query_embedding.reshape(1, -1),
#                 embedding.reshape(1, -1)
#             )[0][0]
#             if similarity > best_similarity and similarity > threshold:
#                 best_match, best_similarity = name, similarity
#     return best_match, best_similarity

# def load_database():
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image = cv2.imread(os.path.join(person_dir, image_filename))
#                     if image is not None:
#                         embeddings.append(extract_embedding(image))
#                 database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# camera_urls = [
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.119:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.135:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
# ]

# database = load_database()

# # Dictionary to store movement trails
# movement_trails = {}

# def process_camera_feed(camera_url, camera_id, frame_queue, shared_data):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open.")
#         return

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break
        
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x, center_y = (x1 + x2) // 2, y2
#                 person_crop = frame[y1:y2, x1:x2]
                
#                 if shared_data['adding_new_person'].value and camera_id == 0:
#                     cv2.putText(frame, f"Capture {shared_data['view_directions'][shared_data['current_view'].value]} View", (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if shared_data['capture_signal'].value:
#                         shared_data['capture_images'].append(person_crop)
#                         print(f"Captured {shared_data['view_directions'][shared_data['current_view'].value]} view.")
#                         shared_data['current_view'].value += 1
#                         shared_data['capture_signal'].value = False
#                         if shared_data['current_view'].value == len(shared_data['view_directions']):
#                             database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                             shared_data['adding_new_person'].value = False
#                             shared_data['capture_images'][:] = []
#                             shared_data['current_view'].value = 0
#                 else:
#                     query_embedding = extract_embedding(person_crop)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"
                    
#                     if camera_id not in movement_trails:
#                         movement_trails[camera_id] = []
#                     movement_trails[camera_id].append((center_x, center_y))
#                     if len(movement_trails[camera_id]) > 50:
#                         movement_trails[camera_id].pop(0)
                    
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)
                    
#                     for i in range(1, len(movement_trails[camera_id])):
#                         cv2.line(frame, movement_trails[camera_id][i - 1], movement_trails[camera_id][i], (255, 0, 0), 2)
        
#         frame_queue.put(cv2.resize(frame, (854, 480)))
#     cap.release()


# if __name__ == "__main__":
#     manager = mp.Manager()
#     shared_data = manager.dict({
#         'adding_new_person': manager.Value('b', False),
#         'capture_signal': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"]
#     })

#     frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
#     processes = [mp.Process(target=process_camera_feed, args=(url, i, frame_queues[i], shared_data), daemon=True) for i, url in enumerate(camera_urls)]
#     for p in processes:
#         p.start()
    
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 cv2.imshow(f"Camera {i}", frame_queues[i].get())
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('n') and not shared_data['adding_new_person'].value:
#             shared_data['adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['adding_new_person'].value = False
#         elif key == ord('c'):
#             shared_data['capture_signal'].value = True
    
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()




#------------------------------------------------------------------------------------------





# import cv2
# import os
# import time
# import torch
# import torch.multiprocessing as mp
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# import logging
# logging.getLogger("ultralytics").setLevel(logging.WARNING)

# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# database_dir = './database'

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
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = reid_model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match, best_similarity = None, 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(
#                 query_embedding.reshape(1, -1),
#                 embedding.reshape(1, -1)
#             )[0][0]
#             if similarity > best_similarity and similarity > threshold:
#                 best_match, best_similarity = name, similarity
#     return best_match, best_similarity

# def load_database():
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image = cv2.imread(os.path.join(person_dir, image_filename))
#                     if image is not None:
#                         embeddings.append(extract_embedding(image))
#                 database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# camera_urls = [
#     # "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.119:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.135:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
# ]

# database = load_database()

# # Dictionary to store movement trails
# movement_trails = {}

# # Example points for ROI (polygon) - Modify as per your needs
# # roi_points = [(300, 470), (720, 460), (470, 250), (240, 320)]  # Example: Rectangular region

# def is_point_in_polygon(point, polygon):
#     """
#     Check if a point is inside a polygon using OpenCV.
#     """
#     return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0

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

# def process_camera_feed(camera_url, camera_id, roi_points, frame_queue, shared_data):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open.")
#         return

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break
        
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x, center_y = (x1 + x2) // 2, y2
#                 person_crop = frame[y1:y2, x1:x2]

#                 if shared_data['adding_new_person'].value and camera_id == 0:
#                     cv2.putText(frame, f"Capture {shared_data['view_directions'][shared_data['current_view'].value]} View", (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if shared_data['capture_signal'].value:
#                         shared_data['capture_images'].append(person_crop)
#                         print(f"Captured {shared_data['view_directions'][shared_data['current_view'].value]} view.")
#                         shared_data['current_view'].value += 1
#                         shared_data['capture_signal'].value = False
#                         if shared_data['current_view'].value == len(shared_data['view_directions']):
#                             database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                             shared_data['adding_new_person'].value = False
#                             shared_data['capture_images'][:] = []
#                             shared_data['current_view'].value = 0

#                 else:
#                     query_embedding = extract_embedding(person_crop)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"

#                     # # Check if the center of the person crosses the ROI
#                     # if is_point_in_polygon((center_x, center_y), roi_points):
#                     #     # The person is inside the ROI
#                     #     cv2.putText(frame, f"{label} Inside ROI ({similarity:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 250, 0), 2)
#                     #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 250, 0), 2)

#                     # else:
#                     #     # The person is outside the ROI
#                     #     cv2.putText(frame, f"{label} Outside ROI ({similarity:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
#                     #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#                     # Check if the person is inside the drawn ROI
#                     if cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False) >= 0:
#                         flabel = f"{label} Inside ROI ({similarity:.2f})"
#                         color = (0, 255, 0)

#                     else:
#                         flabel = f"{label} Outside ROI ({similarity:.2f})"
#                         color = (0, 0, 255)

#                     cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        
#                     # Trigger Alarm if the person exits the region
#                     if camera_id not in movement_trails:
#                         movement_trails[camera_id] = []
#                     movement_trails[camera_id].append((center_x, center_y))

#                     if len(movement_trails[camera_id]) > 30:
#                         movement_trails[camera_id].pop(0)
                    
#                     for i in range(1, len(movement_trails[camera_id])):
#                         cv2.line(frame, movement_trails[camera_id][i - 1], movement_trails[camera_id][i], (255, 0, 0), 2)

#                     # Check if the person crosses the boundary of ROI (exit detection)
#                     if len(movement_trails[camera_id]) > 1:
#                         last_point = movement_trails[camera_id][-2]
#                         current_point = movement_trails[camera_id][-1]
#                         if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
#                             print(f"ALERT: Person {label} entered the ROI.")
#                         elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
#                             print(f"ALERT: Person {label} exited the ROI.")
                
#                     # Draw the bounding box and movement trail
#                     # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#         # Draw the ROI boundary on the frame
#         cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 255), thickness=2)
#         frame_queue.put(cv2.resize(frame, (854, 480)))

#     cap.release()



# if __name__ == "__main__":
#     manager = mp.Manager()
#     shared_data = manager.dict({
#         'adding_new_person': manager.Value('b', False),
#         'capture_signal': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"]
#     })

#     rois = []
#     for i, url in enumerate(camera_urls):
#         cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#         success, frame = cap.read()
#         cap.release()
#         if not success:
#             print(f"Error: Camera {i} failed to capture frame.")
#             continue
#         print(f"Draw ROI for Camera {i}, press 'q' when done.")
#         rois.append(draw_roi(frame))

#     frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
#     processes = [mp.Process(target=process_camera_feed, args=(url, i, rois[i], frame_queues[i], shared_data), daemon=True) for i, url in enumerate(camera_urls)]

#     for p in processes:
#         p.start()
    
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 cv2.imshow(f"Camera {i}", frame_queues[i].get())
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('n') and not shared_data['adding_new_person'].value:
#             shared_data['adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['adding_new_person'].value = False
#         elif key == ord('c'):
#             shared_data['capture_signal'].value = True
    
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()









#------------------ FEB 10











# import cv2
# import os
# import time
# import torch
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from queue import Queue
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# import logging
# logging.getLogger("ultralytics").setLevel(logging.WARNING)

# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_ibn_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# database_dir = './database'

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
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = reid_model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match, best_similarity = None, 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]

#             if similarity > best_similarity and similarity > threshold:
#                 best_match, best_similarity = name, similarity
#     return best_match, best_similarity

# def load_database():
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image = cv2.imread(os.path.join(person_dir, image_filename))
#                     if image is not None:
#                         embeddings.append(extract_embedding(image))
#                 database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# camera_urls = [
#     # "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.119:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.135:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
# ]

# database = load_database()



# def is_point_in_polygon(point, polygon):
#     """
#     Check if a point is inside a polygon using OpenCV.
#     """
#     return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0

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

# def process_camera_feed(camera_url, camera_id, roi_points, frame_queue, shared_data):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open.")
#         return

#     # Dictionary to store movement trails
#     movement_trails = {}
#     frame_count = 0

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break
        
#         # cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0])
#         for result in results:
#             if result.boxes is None:  # Check if no detections exist
#                 continue  

#             track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)

#             # for box in result.boxes:
#             for box, track_id in zip(result.boxes, track_ids):
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x, center_y = (x1 + x2) // 2, y2
#                 person_crop = frame[y1:y2, x1:x2]

#                 track_id = int(track_id) if track_id is not None else -1

#                 if shared_data['adding_new_person'].value and camera_id == 0:
#                     cv2.putText(frame, f"Capture {shared_data['view_directions'][shared_data['current_view'].value]} View", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if shared_data['capture_signal'].value:
#                         shared_data['capture_images'].append(person_crop)
#                         print(f"Captured {shared_data['view_directions'][shared_data['current_view'].value]} view.")
#                         shared_data['current_view'].value += 1
#                         shared_data['capture_signal'].value = False

#                         if shared_data['current_view'].value == len(shared_data['view_directions']):
#                             database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                             shared_data['adding_new_person'].value = False
#                             shared_data['capture_images'][:] = []
#                             shared_data['current_view'].value = 0


#                 elif shared_data['dynamic_adding_new_person'].value and camera_id in [0,1]:
#                     print(f'Camera: {camera_id}, Frame: {frame_count}')
#                     cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
#                     if frame_count%5 == 0:
#                         shared_data['capture_images'].append(person_crop)

#                     elif frame_count >= 200:
#                         database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                         shared_data['dynamic_adding_new_person'].value = False
#                         shared_data['capture_images'][:] = []

                    

#                 else:
#                     query_embedding = extract_embedding(person_crop)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"

#                     # Check if the person is inside the drawn ROI
#                     if cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False) >= 0:
#                         flabel = f"{label} Inside ROI ({similarity:.2f})"
#                         color = (0, 255, 0)

#                     else:
#                         flabel = f"{label} Outside ROI ({similarity:.2f})"
#                         color = (0, 0, 255)

#                     cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        
#                     # Trigger Alarm if the person exits the region
#                     if track_id not in movement_trails:
#                         movement_trails[track_id] = []
#                     movement_trails[track_id].append((center_x, center_y))

#                     if len(movement_trails[track_id]) > 30:
#                         movement_trails[track_id].pop(0)
                    
#                     for i in range(1, len(movement_trails[track_id])):
#                         cv2.line(frame, movement_trails[track_id][i - 1], movement_trails[track_id][i], (255, 0, 0), 2)

#                     # Check if the person crosses the boundary of ROI (exit detection)
#                     if len(movement_trails[track_id]) > 1:
#                         last_point = movement_trails[track_id][-2]
#                         current_point = movement_trails[track_id][-1]

#                         if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
#                             print(f"ALERT: Person {label} entered the ROI.")
#                         elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
#                             print(f"ALERT: Person {label} exited the ROI.")
                
#                     # Draw the bounding box and movement trail
#                     # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
#                 frame_count+=1

#         # Draw the ROI boundary on the frame
#         cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 255), thickness=2)
#         frame_queue.put(cv2.resize(frame, (854, 480)))

#     cap.release()



# if __name__ == "__main__":
#     manager = mp.Manager()
#     shared_data = manager.dict({
#         'dynamic_adding_new_person': manager.Value('b', False),
#         'adding_new_person': manager.Value('b', False),
#         'capture_signal': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"]
#     })

#     rois = []
#     for i, url in enumerate(camera_urls):
#         cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#         success, frame = cap.read()
#         cap.release()
#         if not success:
#             print(f"Error: Camera {i} failed to capture frame.")
#             continue
#         print(f"Draw ROI for Camera {i}, press 'q' when done.")
#         rois.append(draw_roi(frame))

#     frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
#     processes = [mp.Process(target=process_camera_feed, args=(url, i, rois[i], frame_queues[i], shared_data), daemon=True) for i, url in enumerate(camera_urls)]

#     for p in processes:
#         p.start()
    
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 cv2.imshow(f"Camera {i}", frame_queues[i].get())

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#         elif key == ord('n') and not shared_data['adding_new_person'].value:
#             shared_data['adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['adding_new_person'].value = False

#         elif key == ord('c'):
#             shared_data['capture_signal'].value = True
            
#         elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value: 
#             shared_data['dynamic_adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['dynamic_adding_new_person'].value = False

    
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()








#-------------  FEB 11








# import cv2
# import os
# import time
# import torch
# import torchreid
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from ultralytics import YOLO
# from tkinter import Tk, simpledialog

# import logging
# logging.getLogger("ultralytics").setLevel(logging.WARNING)

# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)

# # Check device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"=>> Device Used for Inference: {device}")

# # Initialize YOLO model
# model = YOLO("yolo11n.engine", task='detect')

# database_dir = './database'

# # Initialize ReID model
# reid_model = torchreid.models.build_model(
#     name='osnet_ibn_x1_0',
#     num_classes=1000,
#     pretrained=True
# ).to(device)
# reid_model.eval()

# # Preprocessing function
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

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
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = reid_model(image_tensor)
#     return embedding.squeeze(0).cpu().numpy()

# def match_embedding(query_embedding, database, threshold=0.7):
#     best_match, best_similarity = None, 0
#     for name, embeddings in database.items():
#         for embedding in embeddings:
#             similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]

#             if similarity > best_similarity and similarity > threshold:
#                 best_match, best_similarity = name, similarity
#     return best_match, best_similarity

# def load_database():
#     database = {}
#     if os.path.exists(database_dir):
#         for person_name in os.listdir(database_dir):
#             person_dir = os.path.join(database_dir, person_name)
#             if os.path.isdir(person_dir):
#                 embeddings = []
#                 for image_filename in os.listdir(person_dir):
#                     image = cv2.imread(os.path.join(person_dir, image_filename))
#                     if image is not None:
#                         embeddings.append(extract_embedding(image))
#                 database[person_name] = embeddings
#     return database

# def get_person_name():
#     root = Tk()
#     root.withdraw()
#     name = simpledialog.askstring("Input", "Enter the name of the person:")
#     root.destroy()
#     return name

# def is_point_in_polygon(point, polygon):
#     """
#     Check if a point is inside a polygon using OpenCV.
#     """
#     return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0

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

# def process_camera_feed(camera_url, camera_id, roi_points, frame_queue, shared_data, database):
#     cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     if not cap.isOpened():
#         print(f"Error: Camera {camera_id} failed to open.")
#         return

#     # Dictionary to store movement trails
#     movement_trails = {}
#     frame_count = 0

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print(f"Error: Camera {camera_id} failed to read frame.")
#             break
        
#         # cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
#         results = model.track(frame, stream=True, persist=True, device=device, classes=[0], conf=0.4)
        
#         for result in results:
#             if result.boxes is None:  # Check if no detections exist
#                 continue  

#             track_ids = result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes)

#             # for box in result.boxes:
#             for box, track_id in zip(result.boxes, track_ids):
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x, center_y = (x1 + x2) // 2, y2
#                 person_crop = frame[y1:y2, x1:x2]

#                 track_id = int(track_id) if track_id is not None else -1

#                 if shared_data['adding_new_person'].value and camera_id == 0:
#                     cv2.putText(frame, f"Capture {shared_data['view_directions'][shared_data['current_view'].value]} View", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     if shared_data['capture_signal'].value:
#                         shared_data['capture_images'].append(person_crop)
#                         print(f"Captured {shared_data['view_directions'][shared_data['current_view'].value]} view.")
#                         shared_data['current_view'].value += 1
#                         shared_data['capture_signal'].value = False

#                         if shared_data['current_view'].value == len(shared_data['view_directions']):
#                             database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                             shared_data['adding_new_person'].value = False
#                             shared_data['capture_images'][:] = []
#                             shared_data['current_view'].value = 0


#                 elif shared_data['dynamic_adding_new_person'].value and camera_id in [1,2,3]:
#                     print(f'Camera: {camera_id}, Frame: {frame_count}')
#                     cv2.putText(frame, f"Dynamically-capturing {shared_data['current_name']} from multiple angles", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
#                     cv2.putText(frame, f"{frame_count}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
#                     frame_count+=1

#                     if frame_count%5 == 0:
#                         shared_data['capture_images'].append(person_crop)

#                     elif frame_count >= 200:
#                         database[shared_data['current_name']] = add_person_to_database(shared_data['current_name'], shared_data['capture_images'])
#                         shared_data['dynamic_adding_new_person'].value = False
#                         shared_data['capture_images'][:] = []
#                         frame_count = 0


#                 else:
#                     query_embedding = extract_embedding(person_crop)
#                     match, similarity = match_embedding(query_embedding, database)
#                     label = match if match else "Unknown"

#                     # Check if the person is inside the drawn ROI
#                     if cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False) >= 0:
#                         flabel = f"{label} Inside ROI ({similarity:.2f})"
#                         color = (0, 255, 0)

#                     else:
#                         flabel = f"{label} Outside ROI ({similarity:.2f})"
#                         color = (0, 0, 255)

#                     cv2.putText(frame, flabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#                     cv2.putText(frame, str(track_id), (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        
#                     # Trigger Alarm if the person exits the region
#                     if track_id not in movement_trails:
#                         movement_trails[track_id] = []
#                     movement_trails[track_id].append((center_x, center_y))

#                     if len(movement_trails[track_id]) > 30:
#                         movement_trails[track_id].pop(0)
                    
#                     for i in range(1, len(movement_trails[track_id])):
#                         cv2.line(frame, movement_trails[track_id][i - 1], movement_trails[track_id][i], (255, 0, 0), 2)

#                     # Check if the person crosses the boundary of ROI (exit detection)
#                     if len(movement_trails[track_id]) > 1:
#                         last_point = movement_trails[track_id][-2]
#                         current_point = movement_trails[track_id][-1]

#                         if not is_point_in_polygon(last_point, roi_points) and is_point_in_polygon(current_point, roi_points):
#                             print(f"ALERT: Person {label} entered the ROI.")
#                             # stop_alert()
#                         elif is_point_in_polygon(last_point, roi_points) and not is_point_in_polygon(current_point, roi_points):
#                             print(f"ALERT: Person {label} exited the ROI.")
#                             # play_alert()
                
#                     # Draw the bounding box and movement trail
#                     # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
#                 # frame_count+=1

#         # Draw the ROI boundary on the frame
#         cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 255), thickness=2)
#         frame_queue.put(cv2.resize(frame, (854, 480)))

#     cap.release()

# camera_urls = [
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/101", 
#     "rtsp://admin:Lucky786$1@192.168.1.251:554/Streaming/Channels/201", 
#     # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.119:554/Streaming/Channels/101?transport=tcp", 
#     "rtsp://admin:Lucky786$1@192.168.1.135:554/Streaming/Channels/101?transport=tcp", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/101", 
#     # "rtsp://admin:Lucky786$1@192.168.1.97:554/Streaming/Channels/201", 
# ]


# ### --- MAIN FUNCTION --- ###
# if __name__ == "__main__":
#     manager = mp.Manager()
#     database = manager.dict(load_database())
#     # database = {}
#     shared_data = manager.dict({
#         'dynamic_adding_new_person': manager.Value('b', False),
#         'adding_new_person': manager.Value('b', False),
#         'capture_signal': manager.Value('b', False),
#         'capture_images': manager.list(),
#         'current_view': manager.Value('i', 0),
#         'current_name': "",
#         'view_directions': ["Front", "Back", "Right", "Left"]
#     })

#     rois = []
#     for i, url in enumerate(camera_urls):
#         cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#         success, frame = cap.read()
#         cap.release()
#         if not success:
#             print(f"Error: Camera {i} failed to capture frame.")
#             continue
#         print(f"Draw ROI for Camera {i}, press 'q' when done.")
#         rois.append(draw_roi(frame))

#     frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
#     processes = [mp.Process(target=process_camera_feed, args=(url, i, rois[i], frame_queues[i], shared_data, database), daemon=True) for i, url in enumerate(camera_urls)]

#     for p in processes:
#         p.start()
    
#     while True:
#         for i in range(len(camera_urls)):
#             if not frame_queues[i].empty():
#                 cv2.imshow(f"Camera {i}", frame_queues[i].get())

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#         elif key == ord('e'):
#             database.clear()
#             print("The database library has been cleaned up.")

#         elif key == ord('n') and not shared_data['adding_new_person'].value:
#             shared_data['adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['adding_new_person'].value = False

#         elif key == ord('c'):
#             shared_data['capture_signal'].value = True
            
#         elif key == ord('d') and not shared_data['dynamic_adding_new_person'].value: 
#             shared_data['dynamic_adding_new_person'].value = True
#             shared_data['current_name'] = get_person_name()
#             if not shared_data['current_name']:
#                 shared_data['dynamic_adding_new_person'].value = False

    
#     cv2.destroyAllWindows()
#     for p in processes:
#         p.terminate()
#         p.join()








#========================================= FEB 18 (WITH multiprocessing)











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

logging.getLogger("ultralytics").setLevel(logging.WARNING)
mp.set_start_method("spawn", force=True)

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

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"Error: Camera {camera_id} failed to read frame.")
            break

        # Get detection/tracking results from YOLO.
        results = model.track(frame, stream=True, persist=True, device=device, classes=[0], conf=0.4)
        
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
        "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/101", 
        "rtsp://admin:Lucky786$1@192.168.1.252:554/Streaming/Channels/201", 
        # "rtsp://admin:Lucky786$1@192.168.1.64:554/Streaming/Channels/101?transport=tcp", 
        "rtsp://admin:Lucky786$1@192.168.1.120:554/Streaming/Channels/101?transport=tcp", 
        "rtsp://admin:Lucky786$1@192.168.1.136:554/Streaming/Channels/101?transport=tcp", 
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

    frame_queues = [mp.Queue(maxsize=7) for _ in camera_urls]
    processes = [
        mp.Process(target=process_camera_feed,
                   args=(url, i, rois[i], frame_queues[i], shared_data, database),
                   daemon=True)
        for i, url in enumerate(camera_urls)
    ]

    for p in processes:
        p.start()

    while True:
        for i in range(len(camera_urls)):
            if not frame_queues[i].empty():
                cv2.imshow(f"Camera {i}", frame_queues[i].get())
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

    cv2.destroyAllWindows()
    for p in processes:
        p.terminate()
        p.join()


