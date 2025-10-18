# Import required libraries
import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model
model = YOLO('yolo12n.pt')  # Use a higher-quality model if speed allows (yolo12s, yolo12m) (yolo12n is default)
names = model.names

# Define vertical line's X position for left-right counting
line_x = 500  # Adjust this based on camera frame
offset = 15   # Tolerance for counting

# Track previous center X positions
track_hist = {}

# IN/OUT counters
car_in = car_out = 0
bus_in = bus_out = 0
#truck_in = truck_out = 0

# Open webcam (0 = default camera, 1 = external)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Print actual webcam resolution
print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# Optional: Debugging mouse coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1

    # Optional: skip frames to maintain speed
    if frame_count % 2 != 0:
        continue

    # Detect vehicles (car=2, bus=5, truck=7)
    results = model.track(frame, persist=True, classes=[2, 5, 7], verbose=False)

    if results and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            # Draw detection box and center
            color = (0, 255, 0)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f'{names[cls]} {track_id}', (x1, y1 - 10), 1, 1)

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
                    elif cls == 5: bus_in += 1
                    elif cls == 7: truck_in += 1
                    # ADD TO DATABASE

                # Moving left (OUT)
                elif prev_x > line_x + offset >= curr_x:
                    if cls == 2: car_out += 1
                    elif cls == 5: bus_out += 1
                    elif cls == 7: truck_out += 1
                    #SUBTRACT FROM DATABASE
    
    # Display counters
    cvzone.putTextRect(frame, f'Cars In: {car_in}', (60, 40), 2, 2, colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'Cars Out: {car_out}', (640, 40), 2, 2, colorR=(0, 0, 120))
    #cvzone.putTextRect(frame, f'bus_in: {bus_in}', (60, 90), 2, 2, colorR=(120, 0, 120))
    #cvzone.putTextRect(frame, f'bus_out: {bus_out}', (640, 90), 2, 2, colorR=(0, 120, 120))
    #cvzone.putTextRect(frame, f'truck_in: {truck_in}', (60, 140), 2, 2, colorR=(120, 120, 0))
    #cvzone.putTextRect(frame, f'truck_out: {truck_out}', (640, 140), 2, 2)

    # Draw vertical counting line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

    # Show frame
    cv2.imshow("RGB", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
