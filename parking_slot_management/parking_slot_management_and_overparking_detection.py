import cv2
import pickle
import numpy as np
import tensorflow as tf
import time
import json
import asyncio
from ultralytics import YOLO
import os
from concurrent.futures import ThreadPoolExecutor

print(os.getcwd())

# ================== CONFIG ==================
HISTORY_LEN = 15
CNN_HISTORY_LEN = 10
slot_padding = 1

# =================== CNN Model ===================
model = tf.keras.models.load_model("parking_model.h5")
class_names = ["good_parking", "over_parking"]

def prepare_frame(frame):
    gray = cv2.resize(frame, (128, 128))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    return np.expand_dims(gray, axis=0)

# =================== State Tracking ===================
slotpoints = []
slot_start_time = {}
last_fee = {}
slot_matrices = {}
cnn_history = {}
slot_status_history = {}

# Load slot positions
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

# Precompute perspective matrices
frame_point = np.float32([(0,0),(740,0),(740,700),(0,700)])
for idx, slotpoint in enumerate(slotpoints):
    src = np.float32(slotpoint)
    slot_matrices[idx] = cv2.getPerspectiveTransform(src, frame_point)

# =================== YOLO ===================
model_yolo = YOLO("yolov8m.pt")
target_classes = [2, 7]

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

def detect_car_in_slot(slotpoint, yolo_results, padding=1):
    poly_np = expand_polygon(slotpoint, padding)
    for box in yolo_results[0].boxes:
        car_padding = 4
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in target_classes :
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
            x1 += car_padding; y1 += car_padding
            x2 -= car_padding; y2 -= car_padding
            if x2 <= x1 or y2 <= y1:
                continue  
            car_rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if any(cv2.pointPolygonTest(poly_np, corner, False) >= 0 for corner in car_rect):
                return True
    return False

def DrawLines(frame, slotpoints):
    for idx, slotpoint in enumerate(slotpoints):
        x, y = slotpoint[0]
        cv2.polylines(frame, [np.int32(slotpoint)], True, (0,255,0), 2)
        cv2.putText(frame, f'Slot_{idx+1}', (x+35,y+30), 1, 1.9,(200,0,0), 2)

# =================== Parking Logic ===================
def CheckParkingSpace(frame, dilate_frame, slotpoints, yolo_results):
    slot_info = []
    DrawLines(frame, slotpoints)

    for idx, slotpoint in enumerate(slotpoints):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(slotpoint,np.int32)], 255)
        cropped_dilate = cv2.bitwise_and(dilate_frame, dilate_frame, mask=mask)
        count = cv2.countNonZero(cropped_dilate)

        # YOLO occupancy
        is_car = detect_car_in_slot(slotpoint, yolo_results, slot_padding)
        history = slot_status_history.get(idx, [])
        history.append(is_car)
        if len(history) > HISTORY_LEN: history.pop(0)
        slot_status_history[idx] = history
        occupied = sum(history) > 0

        # Duration + fee
        if occupied:
            status = 'Occupied'
            color = (0,0,255)
            if idx not in slot_start_time:
                slot_start_time[idx] = time.time()
            duration_sec = int(time.time() - slot_start_time[idx])
            fee = 12*duration_sec if duration_sec<=60 else 60*12 + (duration_sec-60)*33
        else:
            if count > 2290:
                status, color = 'Unknown', (0,255,0)
            else:
                status, color = 'Empty', (0,255,0)
            duration_sec, fee = 0, last_fee.get(idx, 0)  # keep last fee
            if idx in slot_start_time: del slot_start_time[idx]

        # CNN check
        cnn_pred = 'good_parking'
        if occupied:
            warped = cv2.warpPerspective(dilate_frame, slot_matrices[idx], (740,700))
            img_array = prepare_frame(warped)
            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            cnn_pred = class_names[pred_class]

            cnn_h = cnn_history.get(idx, [])
            cnn_h.append(cnn_pred)
            if len(cnn_h) > CNN_HISTORY_LEN: cnn_h.pop(0)
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
            if duration_sec>60:
                cv2.putText(frame, "Penalty", (slotpoint[0][0]+5, slotpoint[0][1]+150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        slot_info.append({
            "name": f"Slot {idx+1}",
            "status": status,
            "duration": duration_sec,
            "fee": fee,
            "prev_fee": last_fee.get(idx, 0),
            "penalty": duration_sec>60,
            "wrong_parking": cnn_pred=='over_parking'
        })
        last_fee[idx] = fee

    with open("slot_data.json", "w") as f:
        json.dump(slot_info, f, indent=4)

# =================== Async Tasks ===================
async def frame_grabber(cap, frame_queue: asyncio.Queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            await asyncio.sleep(0.05)
            continue
        frame = cv2.resize(frame, (740,700))
        await frame_queue.put(frame)

async def process_frame(frame_queue: asyncio.Queue, executor):
    loop = asyncio.get_event_loop()

    def run_yolo(frame):
        return model_yolo.predict(frame, conf=0.09, verbose=False)

    while True:
        frame = await frame_queue.get()

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 1)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,25,16)
        median = cv2.medianBlur(thresh,5)
        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.dilate(median, kernel, iterations=2)

        # YOLO async
        yolo_task = loop.run_in_executor(executor, run_yolo, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        yolo_results = await yolo_task

        CheckParkingSpace(frame, dilate, slotpoints, yolo_results)

        cv2.imshow("ESP32-CAM Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            os._exit(0)

        await asyncio.sleep(0.05)

async def main():
    url = "http://192.168.1.13:81/stream"
    cap = cv2.VideoCapture(url)
    frame_queue = asyncio.Queue(maxsize=5)

    executor = ThreadPoolExecutor(max_workers=4)

    await asyncio.gather(
        frame_grabber(cap, frame_queue),
        process_frame(frame_queue, executor)
    )

if __name__ == "__main__":
    asyncio.run(main())
