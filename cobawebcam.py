import cv2
import streamlit as st
from ultralytics import YOLO

# ini diganti ke tempat nyimpem modelnya
model_path = "C:\Users\LENOVO\Documents\Project\Egg-YoloV8\training result\train2\weight\best.pt"

# Setting page layout
st.set_page_config(
    page_title="Egg-YOLOv8",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Image/Video Config")

    video_source = st.radio("Pilih sumber video:", ("Webcam", "Upload Video"))

    if video_source == "Webcam":

        try:
            vid_cap = cv2.VideoCapture(0) 
            st.success("Webcam berhasil diakses")
        except:
            st.error("Tidak dapat mengakses webcam")

    else:
        uploaded_video = st.file_uploader("Upload Video...", type=["mp4", "mov", "avi"])

    confidence = None

st.title("Egg-YoloV8")

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


if video_source == "Webcam":
    st_frame = st.empty()
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            image = cv2.resize(image, (720, int(720*(9/16))))
            res = model.predict(image)
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                          caption='Detected Video (Webcam)',
                          channels="BGR",
                          use_column_width=True)
        else:
            vid_cap.release()
            break

elif uploaded_video:
    video_name = uploaded_video.name
    print(uploaded_video.name)

    video_data = uploaded_video.read()

    st.video(video_data)
    if st.sidebar.button('Mulai deteksi'):
        try:
            vid_cap = cv2.VideoCapture(video_name)
        except:
            print(uploaded_video)
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                                caption='Detected Video (Uploaded)',
                                channels="BGR",
                                use_column_width=True)
            else:
                vid_cap.release()
                break
