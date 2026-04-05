# os import
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# required libraries
import cv2
from ultralytics import YOLO
import cvzone
import time
from threading import Thread

#for saving/sending to databse
import json
from datetime import datetime, timezone

# (optimized frame rate) 
class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720, fps=30):     #CHANGE SRC to 1 or 2 if experiencing errors
        # Initialize video stream parameters
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPEG')) # decive betweeen MJPEG or other
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        
        # Initial read to check if stream is open
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = not self.grabbed

    def start(self):
        if not self.stopped:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            if self.stopped:
                # Release the stream and exit the thread
                self.stream.release()
                return
            
            # Read the next frame immediately and overwrite the buffer
            (self.grabbed, self.frame) = self.stream.read()
            
            # If grabbing a frame fails (e.g., camera disconnected), stop
            if not self.grabbed:
                self.stopped = True

    def read(self):
        # Return the latest frame immediately (non-blocking)
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True

# Load YOLOv8 model
model = YOLO('yolov8n.pt') # Use a higher-quality model if compatible (yolo12s, yolo12m) (yolo12n is default)
names = model.names

# Define vertical line's X position for left-right counting
line_x = 520 # based on camera frame 
offset = 15  

# Track previous center X positions
track_hist = {}

# in/out counters
car_in = car_out = 0

# NOTE: When running camera on Raspberry Pi, POE is connected. The cap must be the following:
# Format for POE Cameras
# url = "CAMERA.INFO"
# cap = cv2.VideoCapture(url)

# Using 1280x720 resolution
stream = WebcamVideoStream(src=0, width=1280, height=720, fps=30).start()
time.sleep(1.0) 

# for firebase format to save events
def save_event_to_file(event_type):
    event = {
        'eventType': event_type,  # "ENTRY" or "EXIT"
        'lotID': 'LotA',  # Change to your lot ID
        'source': 'RASPBERRY_PI5',
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }
    
    # Read existing events
    try:
        with open('events.json', 'r') as f:
            events = json.load(f)
    except FileNotFoundError:
        events = []
    
    # Add new event
    events.append(event)
    
    # Keep only last 100 events
    if len(events) > 100:
        events = events[-100:]
    
    # Save back to file
    with open('events.json', 'w') as f:
        json.dump(events, f, indent=2)


# Optional: Debugging mouse coordinates
#def RGB(event, x, y, flags, param):
#   if event == cv2.EVENT_MOUSEMOVE:
#       print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
#cv2.setMouseCallback("RGB", RGB)

frame_count = 0
current_results = None 


while True:
    frame = stream.read()   #stream will stop if serious error occurs (fps < 1)
    if frame is None or stream.stopped:
        print("Failed to grab frame")
        break

    frame_count += 1

    # Optional: skip frames to maintain speed
    if frame_count % 2 == 0: 
    # if frame_count % 2 != 0:
    #     continue

    # Detect vehicles (car=2)
        results = model.track(frame, persist=True, classes=[2, 5, 7], verbose=False)
        current_results = results 

        # tracking history updating (chatgpt)
        if current_results and current_results[0].boxes.id is not None:
            ids = current_results[0].boxes.id.cpu().numpy().astype(int)
            
            # Memory Leak Prevention: Clean up track history 
            active_ids = set(ids)
            ids_to_remove = [tid for tid in track_hist if tid not in active_ids]
            for tid in ids_to_remove:
                del track_hist[tid]
                
            boxes = current_results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = current_results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, class_ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                
                if track_id not in track_hist:
                    track_hist[track_id] = []
                track_hist[track_id].append(cx)

                # Check if vehicle crossed vertical line
                if len(track_hist[track_id]) >= 2:
                    prev_x = track_hist[track_id][-2]
                    curr_x = track_hist[track_id][-1]

                    # Moving right (IN)
                    if prev_x < line_x - offset <= curr_x:
                        if cls == 2: car_in += 1
                        save_event_to_file('ENTRY')
                        # ADD 1 TO DATABASE

                    # Moving left (OUT)
                    elif prev_x > line_x + offset >= curr_x:
                        if cls == 2: car_out += 1
                        save_event_to_file('EXIT')
                        #SUBTRACT 1 FROM DATABASE
        
    # drawing logic (example code)
    if current_results and current_results[0].boxes.id is not None:
        ids = current_results[0].boxes.id.cpu().numpy().astype(int)
        boxes = current_results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = current_results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            # Draw detection box and center
            color = (0, 255, 0)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f'{names[cls]} {track_id}', (x1, y1 - 10), 1, 1)

    # counter display
    cvzone.putTextRect(frame, f'Cars In: {car_in}', (60, 40), 2, 2, colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'Cars Out: {car_out}', (640, 40), 2, 2, colorR=(0, 0, 120))

    # Draw vertical counting line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

    # Show frame can be removed in final py script
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()
cv2.destroyAllWindows()
