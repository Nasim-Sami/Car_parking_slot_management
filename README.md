# Car_parking_slot_management
<br>
This project is applied using Esp32 Cam and Yolo-V8(for car detection). I made a datset of binary images of good parking and overparking. Then Applied Convolutional Neural Network to classify good parking and overparking. and saved the model as "parking_model.h5". Then I selected my slotpoints for drawing a rectangle on each slot. Call this model named "parking_model.h5" inside parking_slot_management_and_overparking_detection.py . 
</br>
<br> 
step 1: Select your region of interest by selecting points.
</br>
<br> 
N.B : You must select each slot points serially to draw polygon on that slot later
</br>
<br>
Run:  Car_parking_slot_management/Detect_slots/detect_slot_points_ROI.py 
</br>
<br>
Step 2: Run : Car_parking_slot_management/Detect_slots/Draw_Polygon.py 
</br>
<br>
Step 3: Run parking_slot_management/parking_slot_management_and_overparking_detection.py
</br>

N.B : Must download "parking_model.h5" and keep in mind the path where you are trying to run your code. If You might require to change path inside code otherwise. Try to make a folder and copy paste python file mentioned in step 1,2,3 and "parking_model.h5" inside that folder. Download all required module and Place ESP32 cam properly. This Code is written for 4 slots specifically. Thank you.
