import cv2
import numpy as np

# --- Step 1: Video stream ---
url = "http://10.19.1.186:81/stream"   # ESP32-CAM stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

print("Stream connected. Press 'q' to quit.")

# --- Step 2: Color Ranges (HSV) ---
color_ranges = {
    "Red Slot": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "Green Slot": [
        (np.array([35, 100, 100]), np.array([85, 255, 255]))
    ],
    "Yellow Slot": [
        (np.array([20, 100, 100]), np.array([30, 255, 255]))
    ],
    "Blue Slot": [
        (np.array([90, 100, 100]), np.array([130, 255, 255]))
    ]
}

# --- Step 3: Process video frames ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break
    frame = cv2.resize(frame, (740, 700))    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for slot_name, ranges in color_ranges.items():
        mask_total = None

        # Handle multiple HSV ranges (for red)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_frame, lower, upper)
            if mask_total is None:
                mask_total = mask
            else:
                mask_total = cv2.bitwise_or(mask_total, mask)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:  # Set your threshold here
                x, y, w, h = cv2.boundingRect(largest)
                
                # Color for drawing the rectangle
                draw_color = (0, 0, 0)
                if "Red" in slot_name: draw_color = (0, 0, 255)
                elif "Green" in slot_name: draw_color = (0, 255, 0)
                elif "Yellow" in slot_name: draw_color = (0, 255, 255)
                elif "Blue" in slot_name: draw_color = (255, 0, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                cv2.putText(frame, slot_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

    cv2.imshow("Parking Slot Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
