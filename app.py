import cv2
import streamlit as st
from ultralytics import YOLO

# ini diganti ke tempat nyimpem modelnya
model_path = "weights/yolov8n.pt"

# Setting page layout
st.set_page_config(
    page_title="Egg-YOLOv8",  
    page_icon="🤖",     
    layout="wide",      
    initial_sidebar_state="expanded"    
)


with st.sidebar:
    st.header("Image/Video Config") 
    
    uploaded_video = st.sidebar.file_uploader("Upload Video...", type=["mp4", "mov", "avi"])
    
    confidence = None

st.title("Egg-YoloV8")

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
# st.write("Model loaded successfully!")

if uploaded_video:
    video_name = uploaded_video.name

        # Dapatkan data video
    video_data = uploaded_video.read()

        # Tampilkan video
    st.video(video_data)
    if st.sidebar.button('Mulai deteksi'):
        vid_cap = cv2.VideoCapture(uploaded_video)
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=confidence)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
            else:
                vid_cap.release()
                break