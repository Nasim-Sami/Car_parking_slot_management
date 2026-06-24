# 🚗 Car Parking Slot Management & Overparking Detection

A computer-vision system that monitors parking slots in real time using an **ESP32-CAM** feed, **YOLOv8** for vehicle detection, and a custom **CNN classifier** to detect overparking (a car parked outside the lines of its slot, encroaching on neighbors).

## 🎯 What It Does

1. Streams live video from an ESP32-CAM mounted above a parking area
2. Detects vehicles in the frame using YOLOv8
3. Checks which marked slot each vehicle occupies (via perspective-corrected polygons)
4. Classifies each slot's image as **good parking** or **over parking** using a CNN trained on a custom binary-image dataset
5. Tracks slot occupancy, duration, and (optionally) parking fees over time

## 🌟 Key Features

- **Live ESP32-CAM streaming** — no extra camera hardware needed beyond the module
- **YOLOv8 vehicle detection** — filters for car/truck classes only
- **Perspective-corrected slot extraction** — each slot is warped to a fixed top-down view before classification, so the CNN sees a consistent frame regardless of camera angle
- **Custom CNN overparking classifier** — binary classifier (`good_parking` vs `over_parking`) trained on grayscale slot images
- **Interactive slot setup** — click to mark the 4 corners of each slot directly on the live stream, no manual coordinate entry
- **Occupancy + duration tracking** — per-slot history buffers smooth out noisy single-frame detections
- **Configurable for any number of slots** — current setup ships pre-configured for 4 slots, but slot points are stored as a flat list and can be extended

## 🧠 How It Works (Pipeline)

```
ESP32-CAM stream
      │
      ▼
[1] detect_slot_points_ROI.py    → click 4 corners per slot → saves to "carParkPosition" (pickle)
      │
      ▼
[2] Draw_polygon.py              → verify slot polygons render correctly on the live feed
      │
      ▼
[3] CNN_model_for_detecting_over_parking.py → train CNN on good_parking/over_parking dataset → parking_model.h5
      │
      ▼
[4] parking_slot_management_and_overparking_detection.py
      ├── YOLOv8 detects vehicles in the full frame
      ├── Each slot is perspective-warped to a fixed 740×700 view
      ├── CNN classifies the warped slot crop as good/over parking
      └── Smoothed via rolling history (HISTORY_LEN / CNN_HISTORY_LEN) to avoid flicker
```

## 📂 Project Structure

```
Car_parking_slot_management/
├── README.md
├── CNN_overparking.py/
│   └── CNN_model_for_detecting_over_parking.py   # Dataset split + CNN training → parking_model.h5
├── Detect_slots/
│   ├── detect_slot_points_ROI.py                 # Click-to-mark slot corners on the live stream
│   ├── Draw_polygon.py                            # Visualize marked slot polygons + labels
│   └── detect_slot_based_on_color.py              # Alternate color-based slot detection approach
└── parking_slot_management/
    ├── Parking_slot_management(without_parallal_task).py   # Single-threaded variant
    └── parking_slot_management_and_overparking_detection.py # Main pipeline (YOLO + CNN + ESP32 stream)
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy tensorflow scikit-learn ultralytics
```

You also need:
- An **ESP32-CAM** flashed with streaming firmware, reachable at a known IP (e.g. `http://<esp32-ip>:81/stream`)
- A YOLOv8 weights file (`yolov8m.pt`, auto-downloaded by `ultralytics` on first run)
- A labeled dataset of slot images split into `good_parking/` and `over_parking/` folders (not included — see note below)

### Step 1 — Mark slot points

Update the stream URL inside the script to your ESP32-CAM's IP, then run:

```bash
python Detect_slots/detect_slot_points_ROI.py
```

Click the **4 corners of each slot, in order, slot by slot**. Points are saved to a `carParkPosition` pickle file. Press `ESC` to finish.

> ⚠️ Corners must be clicked **serially per slot** (all 4 of slot 1, then all 4 of slot 2, ...) — the pipeline groups points in chunks of 4 in click order.

### Step 2 — Verify slot polygons

```bash
python Detect_slots/Draw_polygon.py
```

Confirms the polygons line up with the physical slots and that labels (`Green_Slot`, `Red_slot`, `Yellow_Slot`, `Blue_Slot`) render correctly.

### Step 3 — Train the overparking CNN

```bash
python "CNN_overparking.py/CNN_model_for_detecting_over_parking.py"
```

This expects a dataset directory with `good_parking/` and `over_parking/` subfolders, does an 80/20 train/test split, trains a grayscale CNN (`128×128` input, 25 epochs), and saves `parking_model.h5`.

> ⚠️ **Dataset not included** — the ~2000-image dataset used to train the classifier was too large to upload. You'll need to build your own (sample images of correctly vs. incorrectly parked vehicles per slot) or request it from the author. Update `BASE_DIR` in the script to point at your dataset.

### Step 4 — Run live detection

```bash
python parking_slot_management/parking_slot_management_and_overparking_detection.py
```

This loads `parking_model.h5` and the saved slot points, streams from the ESP32-CAM, runs YOLOv8 detection, warps each slot to a fixed perspective, classifies it, and overlays live status on the video feed.

## ⚙️ Configuration Notes

- **Hardcoded for 4 slots** — extending to more requires updating `slot_name` labels in `Draw_polygon.py` and re-marking points
- **Paths are local to the original dev machine** — `BASE_DIR` in the CNN trainer and the ESP32 stream URLs in each script are hardcoded; update them for your setup before running
- **History smoothing** — `HISTORY_LEN` (slot detection) and `CNN_HISTORY_LEN` (classification) control how many recent frames are averaged before a status change is reported, reducing false flicker from single bad frames

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Vehicle detection | YOLOv8 (`ultralytics`) |
| Overparking classification | Custom CNN (TensorFlow/Keras) |
| Video capture | OpenCV + ESP32-CAM MJPEG stream |
| Slot geometry | Perspective transform (`cv2.getPerspectiveTransform`) |
| Slot point persistence | Python `pickle` |

## 📄 License

Academic / hackathon project — educational use.
