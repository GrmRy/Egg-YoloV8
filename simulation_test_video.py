from ultralytics import YOLO
import cv2

#import model
model = YOLO('training result/train2/weight/best.pt')

#open video with opencv
video = 'video\VID_20240102_191221.mp4'
cap = cv2.VideoCapture(video)

while cap.isOpened():
    succes, frame = cap.read()

    if succes:
        frame = cv2.resize(frame, (720,480))
        results = model(frame, classes=0)

        detect_frame = results[0].plot()

        #display the frame
        cv2.imshow("show", detect_frame)

        if cv2.waitKey(24) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
