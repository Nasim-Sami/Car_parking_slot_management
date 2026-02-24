import cv2
import pickle
import numpy as np

try:
    with open('carParkPosition', 'rb') as f:
        positionList = pickle.load(f)
        print(positionList)
        # positionList.remove((609, 459))
        positionList.remove((211, 241))
        positionList.pop()

except:
    positionList = []



def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        positionList.append((x, y))
        print("Added:", (x, y))   # ✅ print only once when clicked
        
        with open('carParkPosition', 'wb') as f:
            pickle.dump(positionList, f)



# ESP32-CAM stream (replace with your camera IP)
url = "http://10.19.129.97:81/stream"   
cap = cv2.VideoCapture(url)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to get frame from ESP32-CAM")
        break

    frame = cv2.resize(frame, (740, 700))
    for position in positionList:
        (x, y) = position
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        

    # DrawLines(frame,slotpoints)
    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', mouseClick)
    


    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
