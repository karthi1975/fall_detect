import cv2
import cvzone
import math
from ultralytics import YOLO
import time

# Open webcam feed (change '0' to the correct camera ID if needed)
cap = cv2.VideoCapture(0)  # For live camera feed

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Set the width and height of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG codec (works well across platforms)
out = cv2.VideoWriter('output_fall_detection_with_bbox.mp4', fourcc, 20.0, (frame_width, frame_height))

# Load YOLO model for fall detection
model = YOLO('yolov8s.pt')

# Load class names from a file (assuming person detection)
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Start time to limit recording to 15 seconds
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    # If the frame is not captured correctly, exit
    if not ret:
        print("Error: Failed to capture video frame.")
        break

    # Perform detection
    results = model(frame)

    # Iterate through each detection result
    for result in results:
        for box in result.boxes:
            # Extract box parameters
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract confidence and class index
            confidence = box.conf[0]
            class_index = int(box.cls[0])
            
            # Get class name and confidence level
            class_name = classnames[class_index]
            conf = math.ceil(confidence * 100)

            # Check for person detection and apply fall detection
            if conf > 80 and class_name == 'person':
                width = x2 - x1
                height = y2 - y1
                threshold = height - width  # Fall detection heuristic

                # Draw bounding box and label on the frame
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_name} {conf}%', [x1, y1 - 10], scale=1.5, thickness=2)

                # Check if fall is detected
                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 30], scale=1.5, thickness=2)

    # Show the frame with bounding boxes and labels
    cv2.imshow('Live Feed', frame)

    # Write the frame to the output file
    out.write(frame)

    # Break the loop after 15 seconds
    if time.time() - start_time > 15:
        break

    # Option to manually stop the recording (press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and file objects
cap.release()
out.release()
cv2.destroyAllWindows()