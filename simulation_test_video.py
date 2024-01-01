from ultralytics import YOLO
import cv2

#import model
model = YOLO('detect/weights/best.pt')

#open video with opencv
video = 'test3.mp4'
cap = cv2.VideoCapture(video)

while cap.isOpened():
    succes, frame = cap.read()

    if succes:
        # frame = cv2.resize(frame, (200,450))
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
