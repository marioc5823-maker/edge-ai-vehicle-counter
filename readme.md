**Edge AI Vehicle Counter:**

Real-time vehicle detection and directional counting pipeline using YOLOv8, deployed on edge hardware (Raspberry Pi). Uses a vertical line crossing algorithm to count vehicles moving IN and OUT of a monitored area.

**How it works:**

1. Webcam captures frames without blocking the main loop
2. YOLOv8/Yolov12 runs inference on every other frame to balance speed and accuracy
3. Each detected vehicle's center point is tracked across frames
4. When a vehicle's trajectory crosses the vertical line it is counted as IN (left → right) or OUT (right → left)
5. Each crossing event is logged to a JSON file in a Firebase-compatible format


**Hardware:**

-Raspberry Pi 4/Raspberry Pi 5

-External USB Camera



**Tech Stack:**

-Python 3.10+

-YOLOv8 (Ultralytics)

-OpenCV

-cvzone



**Demo:**

//enter URL
