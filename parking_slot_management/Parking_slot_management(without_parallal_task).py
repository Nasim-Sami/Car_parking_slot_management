import cv2
import pickle
import numpy as np
import tensorflow as tf
import time
import json
from ultralytics import YOLO
import os
print(os.getcwd())

# ================== CONFIG ==================
HISTORY_LEN = 15  # frames to smooth occupancy
CNN_HISTORY_LEN = 10

slot_padding = 1

# =================== CNN Model ===================
model = tf.keras.models.load_model("parking_model.h5")
class_names = ["good_parking", "over_parking"]

def prepare_frame(frame):
    gray = cv2.resize(frame, (128, 128))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)  # shape (128,128,1)
    return np.expand_dims(gray, axis=0)   # shape (1,128,128,1)

# =================== Load Slots ===================
slotpoints = []
slot_start_time = {}
last_fee = {}
slot_matrices = {}  # cached perspective matrices
cnn_history = {}     # rolling window for CNN predictions
slot_status_history = {}  # rolling window for YOLO detection

try:
    with open('carParkPosition', 'rb') as f:
        positionList = pickle.load(f)
except:
    positionList = []

for i in range(0, len(positionList), 4):
    points = []
    for j in range(4):
        try:
            points.append(positionList[i+j])
        except IndexError:
            break
    if len(points) == 4:
        slotpoints.append(points)

# Precompute perspective matrices for CNN warp
frame_point = np.float32([(0,0),(740,0),(740,700),(0,700)])
for idx, slotpoint in enumerate(slotpoints):
    src = np.float32(slotpoint)
    slot_matrices[idx] = cv2.getPerspectiveTransform(src, frame_point)

def DrawLines(frame, slotpoints):
    for idx, slotpoint in enumerate(slotpoints):
        x, y = slotpoint[0]
        cv2.polylines(frame, [np.int32(slotpoint)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(frame, f'Slot_{idx+1}', (x+35,y+30), 1, 1.9,(200,0,0), thickness=2)

# =================== YOLO ===================

model_yolo = YOLO("yolov8m.pt")
target_classes = [2, 7]  # car and truck

def expand_polygon(poly_points, padding):
    poly_np = np.array(poly_points, dtype=np.float32)
    cx, cy = np.mean(poly_np[:,0]), np.mean(poly_np[:,1])
    expanded = []
    for x, y in poly_np:
        vec_x, vec_y = x - cx, y - cy
        length = np.sqrt(vec_x**2 + vec_y**2)
        if length != 0:
            vec_x, vec_y = vec_x/length, vec_y/length
        expanded.append([int(x + vec_x*padding), int(y + vec_y*padding)])
    return np.array(expanded, dtype=np.int32)

def detect_car_in_slot(slotpoint, yolo_results, padding = 1):
    """
    Check if any detected car overlaps with the slot
    """
    poly_np = expand_polygon(slotpoint, padding)
    for box in yolo_results[0].boxes:
        car_padding = 4
        cls_id = int(box.cls[0])
        if cls_id in target_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
             # --- contract bounding box directly here ---
            x1 += car_padding
            y1 += car_padding
            x2 -= car_padding
            y2 -= car_padding

            # prevent invalid rect
            if x2 <= x1 or y2 <= y1:
                continue  

            car_rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
           
            if any(cv2.pointPolygonTest(poly_np, corner, False) >= 0 for corner in car_rect):
                return True
    return False

# =================== Main Parking Check ===================
from concurrent.futures import ThreadPoolExecutor

def process_slot(idx, slotpoint, frame, dilate_frame, yolo_results):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask,[np.array(slotpoint,np.int32)], 255)
    cropped_dilate = cv2.bitwise_and(dilate_frame, dilate_frame, mask=mask)
    count = cv2.countNonZero(cropped_dilate)

    # YOLO occupancy check
    is_car = detect_car_in_slot(slotpoint, yolo_results, slot_padding)
    history = slot_status_history.get(idx, [])
    history.append(is_car)
    if len(history) > HISTORY_LEN:
        history.pop(0)
    slot_status_history[idx] = history
    occupied = sum(history) > HISTORY_LEN // 2  # smoother

    # Duration & fee
    if occupied:
        status = 'Occupied'
        if idx not in slot_start_time:
            slot_start_time[idx] = time.time()
        duration_sec = int(time.time() - slot_start_time[idx])
        fee = 12*duration_sec if duration_sec<=60 else 60*12 + (duration_sec-60)*33
    else:
        if count > 2290:
            status = 'Unknown'
        else:
            status = 'Empty'
        duration_sec, fee = 0, 0
        if idx in slot_start_time:
            del slot_start_time[idx]

    # CNN over-parking
    cnn_pred = 'good_parking'
    if occupied:
        warped = cv2.warpPerspective(dilate_frame, slot_matrices[idx], (740,700))
        img_array = prepare_frame(warped)
        pred = model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        cnn_pred = class_names[pred_class]

        cnn_h = cnn_history.get(idx, [])
        cnn_h.append(cnn_pred)
        if len(cnn_h) > CNN_HISTORY_LEN:
            cnn_h.pop(0)
        cnn_history[idx] = cnn_h

        if cnn_h.count('over_parking') > CNN_HISTORY_LEN//2:
            cv2.putText(frame, "Wrong_parking", (slotpoint[0][0]+5, slotpoint[0][1]+120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Draw info
    cv2.putText(frame, f"{status} ({count})", (slotpoint[0][0]+5, slotpoint[0][1]+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,160,130), 2)
    if occupied:
        cv2.putText(frame, f"{duration_sec}s, Tk:{fee}", (slotpoint[0][0]+5, slotpoint[0][1]+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    return {
        "name": f"Slot {idx+1}",
        "status": status,
        "duration": duration_sec,
        "fee": fee,
        "prev_fee": last_fee.get(idx, 0),
        "penalty": duration_sec>60,
        "wrong_parking": cnn_pred=='over_parking'
    }

def CheckParkingSpaceParallel(frame, dilate_frame, slotpoints, yolo_results):
    DrawLines(frame, slotpoints)
    slot_info = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_slot, idx, sp, frame, dilate_frame, yolo_results)
                   for idx, sp in enumerate(slotpoints)]
        for f in futures:
            slot_info.append(f.result())

    for info in slot_info:
        last_fee[int(info["name"].split()[-1])-1] = info["fee"]

    with open("slot_data.json", "w") as f:
        json.dump(slot_info, f, indent=4)

    cv2.imshow("Parking Slots Status", frame)


# =================== Stream ===================
url = "http://10.19.129.97:81/stream"
cap = cv2.VideoCapture(url)

while True:
    start_time = time.time()
    results = []

    while (time.time() - start_time) < 5:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            continue

        frame = cv2.resize(frame, (740,700))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 1)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,25,16)
        median = cv2.medianBlur(thresh,5)
        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.dilate(median, kernel, iterations=2)

        # Run YOLO once per frame
        yolo_results = model_yolo.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.07)

        CheckParkingSpaceParallel(frame, dilate, slotpoints, yolo_results)


        cv2.imshow("ESP32-CAM Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        time.sleep(0.2)

    # CNN majority voting over 5 sec
    good_count = sum([1 for idx in cnn_history if cnn_history[idx][-1]=='good_parking'])
    bad_count = sum([1 for idx in cnn_history if cnn_history[idx][-1]=='over_parking'])
    total = good_count + bad_count
    if total>0:
        good_ratio = good_count/total
        print("✅ GOOD PARKING" if good_ratio>=0.59 else "❌ BAD PARKING")
    print(f"Cycle finished with {total} slots analyzed.\n🔄 Starting next 5-sec cycle...\n")
