from ultralytics import YOLO
import cv2

#import model
model = YOLO('yolov8n.pt')

#open video with opencv
video = ''
cap = cv2.VideoCapture(video)

while cap.isOpened():
    succes, frame = cap.read()
    results = model(frame, conf=0.7, classes=0)
    