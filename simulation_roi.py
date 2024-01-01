from ultralytics import YOLO
import cv2
import numpy as np

#import model
model = YOLO('yolov8n.pt')

#open video with opencv
video = ''
cap = cv2.VideoCapture(video)

## roi coordinates
coordinates = [[]]
egg_detected = []

while cap.isOpened():
    succes, frame = cap.read()

    if succes:
        results = model(frame, conf=0.7, classes=0)

        # detect_frame = results[0].plot()
        for result in results:
            boxes = result.boxes.data.to('cpu').numpy().astype(int)
            for box in boxes:
                x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

                #center point
                cx = int(x1 + x2)//2
                cy = int(y2 + y1)//2

                #polygon test roi
                for i, area_coords in enumerate(coordinates):
                    result_pol =cv2.pointPolygonTest(np.array(area_coords, np.int32), (cx, cy), False)

                    if result_pol >=0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        egg_detected[i] = 1

        egg_count = egg_detected.count(1)
        cv2.putText(frame, f"All eggs: {len(results[0])}", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
                    
        #display the frame
        # cv2.imshow("show", detect_frame)

        if cv2.waitKey(24) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
