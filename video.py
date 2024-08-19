import cv2
import math
import random
from ultralytics import YOLO

path_model = "model.pt"
model = YOLO(path_model)

vide_path ="../../../../Videos/Captures/Terabet — Mozilla Firefox 2024-08-13 18-11-59.mp4"

colors = {}
for class_id in range(len(model.names)):
    colors[class_id] = [random.randint(0, 255) for _ in range(3)]

def calculate_distance(box1, box2):
    x1, y1 = (box1[0] + box2[2]) / 2, (box1[1] + box2[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

cap = cv2.VideoCapture(vide_path)
if not cap.isOpened():
    print("Error opening video stream")
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    height, width, _ = frame.shape

    x_start = int(width * 0.2) 
    x_end = int(width * 0.8)   
    y_start = int(height * 0.1)
    y_end = int(height * 0.9)   

    frame_cropped = frame[y_start:y_end, x_start:x_end]
    
    frame_resized = cv2.resize(frame_cropped, (640, 640))


    result = model.predict(frame_resized, save=False, conf=0.4, iou=0.7)


    for box in result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        x1 = int(x1 * (x_end - x_start) / 640) + x_start
        y1 = int(y1 * (y_end - y_start) / 640) + y_start
        x2 = int(x2 * (x_end - x_start) / 640) + x_start
        y2 = int(y2 * (y_end - y_start) / 640) + y_start

        class_id = int(box.cls[0])
        color = colors[class_id]

        # label = f"{result[0].names[class_id]} {box.conf[0]:.2f}"
        label = f"{result[0].names[class_id]}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
  
    
    # result_img = result[0].plot()
    cv2.imshow("Real-time detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()