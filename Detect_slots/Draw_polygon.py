import cv2
import pickle
import numpy as np

try:
    with open('carParkPosition', 'rb') as f:
        positionList = pickle.load(f)
        print(positionList)
        

except:
    positionList = []

slotpoints = []
for i in range(0, len(positionList), 4):
    points = []
    for j in range(4):
        try:
            points.append(positionList[i+j])
        except IndexError:
            break
    if len(points) == 4:
        slotpoints.append(points)

def DrawLines(frame,slotpoints):
    slot_name=['Green_Slot','Red_slot','Yellow_Slot','Blue_Slot']
    for idx, slotpoint in enumerate(slotpoints):
        src = np.float32(slotpoint)
    #  draw polygon on original frame
        cv2.polylines(frame, [np.int32(src)], isClosed=True, color=(0,255,0), thickness=2)
        # if idx == 0 :
        #     cv2.putText(frame,'Green_Slot',(166,167), 1, 1.9,(200,0,0),thickness=2)
        # else :
        cv2.putText(frame,slot_name[idx],slotpoint[3], 1, 1.3,(200,0,0),thickness=2)


# ESP32-CAM stream (replace with your camera IP)
url = "http://10.19.1.200:81/stream"   # <-- change to your ESP32-CAM stream URL
cap = cv2.VideoCapture(url)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to get frame from ESP32-CAM")
        break

    frame = cv2.resize(frame, (740, 700))

    DrawLines(frame,slotpoints)
    cv2.imshow('frame', frame)

    


    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
