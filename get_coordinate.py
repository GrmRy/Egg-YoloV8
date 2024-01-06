import cv2
import numpy as np

video = 'video\VID_20240102_193248.mp4'

def rgb(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsrgb = [x,y]
        print(colorsrgb)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', rgb)

area = [[145,1],[230,1],[230,479],[145,479]]
area2= [[269,325],[635,325],[620,305],[279,305]]

cap = cv2.VideoCapture(video)
while cap.isOpened():
    succes, frame = cap.read()
    if succes:
        frame = cv2.resize(frame, (720,480))
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(1,1,1),2)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(24) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
