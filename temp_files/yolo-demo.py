import cv2

# Replace with your RTSP URL
# rtsp_url = "rtsp://foscam@123:foscam@123@192.168.1.104:88/videoMain"
rtsp_url = "rtsp://admin:Lucky786%241@192.168.1.108:554/Streaming/Channels/101"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream. Check the URL or camera configuration.")
else:
    print("RTSP stream opened successfully.")

# Process the video feed frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame. Exiting...")
        break

    # Display the frame
    cv2.imshow("RTSP Video Feed", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()

