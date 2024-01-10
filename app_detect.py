import streamlit as st
import tempfile

import cv2
import cvzone
from ultralytics import YOLO
import numpy as np


# Change this into the right path of the model
model_path = "training result/train2/weight/best.pt"

# Setting page layout
st.set_page_config(
    page_title="Egg-YOLOv8 Detection",  
    page_icon="🤖",     
    layout="wide",      
    initial_sidebar_state="expanded"    
)

with st.sidebar:
    st.header("Image/Video Config")
    video_source = st.radio("Pilih sumber video:", ("Webcam", "Upload Video"))
    if video_source == "Webcam":
        try:
            vid_cap = cv2.VideoCapture(0) 
            st.success("Webcam berhasil diakses!", icon="✅")
        except:
            st.error("Tidak dapat mengakses webcam")
    else:
        uploaded = st.file_uploader("Upload Video...", type=["mp4", "mov", "avi"])

st.title("Egg-YoloV8 Detection")


try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


if video_source == "Webcam":
    if st.sidebar.button('Mulai Deteksi'):
        try:
            vid_cap = cv2.VideoCapture(0)
        except:
            print("error")

        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, frame = vid_cap.read()
            if success:
                frame = cv2.resize(frame, (720,480))
                results = model.predict(frame, conf=0.8, classes=0)
                for result in results:
                    boxes = result.boxes.data.to('cpu').numpy().astype(int)
                    for box in boxes:
                        x1,y1,x2,y2, id = box[0], box[1], box[2], box[3], box[4]
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        
                cv2.putText(frame, f"All eggs: {len(results[0])}", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)         
                #display the frame
                st_frame.image(frame,
                                caption='Detected Video (Uploaded)',
                                channels="BGR",
                                use_column_width=True)
            else:
                vid_cap.release()
                break

elif uploaded:
    uploaded_video = tempfile.NamedTemporaryFile(delete=False) 
    uploaded_video.write(uploaded.read())
    video_name = uploaded_video.name
    if st.sidebar.button('Mulai Deteksi'):
        try:
            vid_cap = cv2.VideoCapture(video_name)
        except:
            print(uploaded_video)

        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, frame = vid_cap.read()
            if success:
                frame = cv2.resize(frame, (720,480))
                results = model.track(frame, conf=0.8, classes=0)
                for result in results:
                    boxes = result.boxes.data.to('cpu').numpy().astype(int)
                    for box in boxes:
                        x1,y1,x2,y2, id = box[0], box[1], box[2], box[3], box[4]
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame, f"All eggs: {len(results[0])}", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)         
                #display the frame
                st_frame.image(frame,
                                caption='Detected Video (Uploaded)',
                                channels="BGR",
                                use_column_width=True)
            else:
                vid_cap.release()
                break