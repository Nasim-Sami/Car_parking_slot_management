# Car_parking_slot_management
<br>
This project uses Esp32 Cam and Yolo-V8(for car detection). I made a datset of binary images of good parking and overparking. Then Applied Convolutional Neural Network to classify good parking and overparking. After that, I saved the model as "parking_model.h5". Then, I selected my slotpoints for drawing a rectangle on each slot. Finally, Called this model named "parking_model.h5" inside parking_slot_management_and_overparking_detection.py to detect if cars are parked properly inside the rectangles(i.e. inside the slots). 
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
step 3 : Run : CNN_overparking.py/CNN_model_for_detecting_over_parking.py
</br>
<br>
Step 4: Run parking_slot_management/parking_slot_management_and_overparking_detection.py
</br>
<br>
N.B : Opps!! Unfortunately Parking Dataset used for classifying good parking and overparking couldn't be uploaded because it is a big file of images (almost 2000 images). Be cautious about the path where you are trying to run your code.  You might require to change path inside code otherwise. Try to make a folder and copy paste all python files mentioned in step 1,2,3 inside that folder. Download all required module and Place ESP32 cam properly. This Code is written for 4 slots specifically. Thank you.
</br>
