# Import necessary libraries
import cv2
from ultralytics import YOLO
import winsound
import pandas as pd
from datetime import datetime
import os

# --- SETTINGS ---
CAMERA_INDEX = 1
MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
LOG_FILE = 'detection_log.csv'
# --- END OF SETTINGS ---

# --- INITIALIZATION ---
# Load the YOLOv8 model
model = YOLO(MODEL_PATH)
class_names = model.names

# Open the webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    exit()

# List to store all detection records for the entire session
session_log = []

print("Starting real-time detection... Press 'q' to quit.")

# --- MAIN LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame. Exiting...")
        break

    # Run YOLOv8 tracking
    results = model.track(frame, persist=True)

    try:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # --- LOGGING EVERY DETECTION ---
            # Get current timestamp for each detection
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # With milliseconds
            class_name = class_names[cls]
            
            # Add the record to our session log
            session_log.append([timestamp, class_name, track_id, conf])

            # Play alert sound for a 'person'
            if class_name == 'person':
                winsound.Beep(440, 100)

            # --- DRAWING ON FRAME ---
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"ID:{track_id} {class_name} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except (AttributeError, IndexError):
        # Handles frames with no detections
        pass

    cv2.imshow("Object Detection Logger", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- SAVING TO SPREADSHEET ---
print("Quitting and saving detection log...")

if session_log: # Only save if something was detected
    # Create a DataFrame from the session's log
    new_df = pd.DataFrame(session_log, columns=['Timestamp', 'Object_Type', 'Tracker_ID', 'Confidence'])

    # Append to existing file or create a new one
    if os.path.exists(LOG_FILE):
        new_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        new_df.to_csv(LOG_FILE, index=False)

    print(f"Log updated in {LOG_FILE}")
else:
    print("No objects were detected to log.")


# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()