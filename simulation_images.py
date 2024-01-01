import cv2
from ultralytics import YOLO

model = YOLO('detect/weights/best.pt')
image = cv2.imread('assets/test3.jpeg')
results = model.predict(image, conf=0.7)
cv2.putText(image, f"All eggs: {len(results[0])}", (10, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

cv2.imshow("frame",results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
