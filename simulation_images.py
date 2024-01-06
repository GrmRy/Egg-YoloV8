import cv2
from ultralytics import YOLO

model = YOLO('training result/train2/weight/best.pt')
image = cv2.imread('assets/test4.jpeg')
results = model.predict(image, conf=0.5) #add argument save=True if you want to save image predict result
cv2.putText(image, f"All eggs: {len(results[0])}", (10, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

cv2.imshow("frame",results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
