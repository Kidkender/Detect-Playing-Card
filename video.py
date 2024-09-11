import cv2
import math
import random
from ultralytics import YOLO
import mss
import numpy as np 
import websockets
import json
import asyncio

path_model = "model.pt"
model = YOLO(path_model)

video_path =""

colors = {}
for class_id in range(len(model.names)):
    colors[class_id] = [random.randint(0, 255) for _ in range(3)]

def calculate_distance(box1, box2):
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

async def send_data(data):
    print("Data to send:", data)
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(data))

async def main():
    created_window = False
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        
        if not created_window:
            cv2.namedWindow("Real-time detection")
            created_window = True
        cv2.namedWindow('Real-time detection', cv2.WINDOW_NORMAL)
        
        while True:
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR) 
            
            height, width, _ = frame.shape

            x_start = int(width * 0.2) 
            x_end = int(width * 0.8)   
            y_start = int(height * 0.1)
            y_end = int(height * 0.9)   
            opponent_threshold = int(height * 0.2)  # 20% from the top


            frame_cropped = frame[y_start:y_end, x_start:x_end]
            frame_resized = cv2.resize(frame_cropped, (640, 640))

            result = model.predict(frame_resized, save=False, conf=0.4, iou=0.5)

            detected_boxes = []
            
            # Populate the all_card dictionary
            data_to_send = {
                "all_card": list(result[0].names[int(box.cls[0])] for box in result[0].boxes),
                "card_in_hand": {},
                "card_in_table": [],
                "card_opponent": []
            }
            
            for box in result[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # x1 = int(x1 * (x_end - x_start) / 640) + x_start
                # y1 = int(y1 * (y_end - y_start) / 640) + y_start
                # x2 = int(x2 * (x_end - x_start) / 640) + x_start
                # y2 = int(y2 * (y_end - y_start) / 640) + y_start

                detected_boxes.append([x1, y1, x2, y2, int(box.cls[0])])
                class_id = int(box.cls[0])
                color = colors[class_id]

            if len(detected_boxes) >= 2:
                min_distance = float('inf')
                closest_pair = None
                
                for i in range(len(detected_boxes)):
                    for j in range(i + 1, len(detected_boxes)):
                        distance = calculate_distance(detected_boxes[i], detected_boxes[j])
                        if distance < min_distance:
                            min_distance = distance
                            closest_pair = (detected_boxes[i], detected_boxes[j])
                
                for box in detected_boxes:
                    x1, y1, x2, y2, class_id = box 
                    color = colors[class_id]
                    
                    label = f"{result[0].names[class_id]}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
                # detected_boxes is array of objects detected
                # this is box: [204, 380, 215, 411, 10] 
                # Index 4 is class_id
                
                cards_on_table_filtered = [
                    card for card in detected_boxes 
                    if card[1] > opponent_threshold  # Ensure the top-left corner (y1) is below the threshold
                ]
                
                # Position of opponent is top, get cards 
                cards_opponent = [
                    result[0].names[card[4]] for card in detected_boxes
                    if card[1] <= opponent_threshold
                ]
                
                if closest_pair:
                    card1, card2 = closest_pair
                    name_card1 = result[0].names[card1[4]]
                    name_card2 = result[0].names[card2[4]]
                    data_to_send["card_in_hand"] = {
                        "card_1": name_card1,
                        "card_2": name_card2
                    }

                    for box in closest_pair:
                        x1, y1, x2, y2, class_id = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # result is Boxes object
                # names is all class 
                # result[0].names[int(box[4])] to get class name
                all_cards_set = set(result[0].names[int(box[4])] for box in cards_on_table_filtered)
                cards_in_hand_set = set(data_to_send["card_in_hand"].values())
                data_to_send["card_in_table"] = list(all_cards_set - cards_in_hand_set)
                data_to_send["card_opponent"] = cards_opponent
            await send_data(data_to_send)

            # Uncomment if you want to see the video output
            # cv2.imshow("Real-time detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
