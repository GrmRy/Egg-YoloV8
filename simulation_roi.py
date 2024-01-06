from ultralytics import YOLO
import cv2
import numpy as np
import cvzone

#import model
model = YOLO('training result/train2/weight/best.pt')

#open video with opencv
video = 'video\VID_20240102_193248.mp4'
cap = cv2.VideoCapture(video)

## roi coordinates vertikal
# coordinates = [[200,1],[230,1],[230,479],[200,479]] 
# roi ccoordinates horizontal
coordinates=[[269,325],[635,325],[620,305],[279,305]]

egg_detected = []


while cap.isOpened():
    succes, frame = cap.read()

    if succes:
        frame = cv2.resize(frame, (720,480))
        results = model.track(frame, conf=0.9, classes=0)

        # detect_frame = results[0].plot()
        for result in results:
            boxes = result.boxes.data.to('cpu').numpy().astype(int)
            for box in boxes:
                x1,y1,x2,y2, id = box[0], box[1], box[2], box[3], box[4]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(frame,str(id), (x1,y1),1,2)
                #center point
                cx = int(x1)
                cy = int(y2)

                #polygon test roi
                test_roi = cv2.pointPolygonTest(np.array(coordinates, np.int32),((cx,cy)), False)
                print(test_roi)
                if test_roi >=0:
                    cv2.circle(frame,(x1,y2),3,(0,0,255),-1)
                    egg_detected.append(id)

        
        cv2.putText(frame, f"All eggs: {len(egg_detected)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
                    
        #display the frame
        cv2.polylines(frame,[np.array(coordinates,np.int32)],True,(1,1,1),2)
        cv2.imshow("show", frame)
        

        if cv2.waitKey(24) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
