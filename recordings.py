import cv2
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
out = cv2.VideoWriter('output_proper_fall_detection.mp4', fourcc, 20.0, (frame_width, frame_height))

# Start time to limit recording to 15 seconds
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    # If the frame is not captured correctly, exit
    if not ret:
        print("Error: Failed to capture video frame.")
        break

    # Show the frame (optional)
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