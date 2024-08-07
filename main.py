import streamlit as st
import tempfile
from yolov10.ultralytics import YOLOv10
import pandas as pd
from io import BytesIO
import cv2
from PIL import Image


def predict(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

    TRAINED_MODEL_PATH = 'best.pt'
    model = YOLOv10('best.pt')
    results = model.predict(source=temp_file_path,
                            imgsz=640)
    annotated_img = results[0].plot()
    return annotated_img


st.title("Helmet Safety Detection")
st.header("Author: @thanhkhanh")
sample = st.button("Test")
st.divider()

col1, col2 = st.columns(2)
file = col1.file_uploader("Upload an image file for helmet detection")


check = col1.button("Check", type="primary")

if check and file is not None:
    pre = predict(file)
    pre1 = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    col2.image(pre1, caption='Detected Image')
    st.balloons()
    st.success('This is a success message!', icon="✅")


elif check and file is None:
    st.warning("Please upload an image first.")

if sample:
    pt = cv2.imread('Black-Workers-Need-a-Bill-of-Rights.jpeg')
    TRAINED_MODEL_PATH = 'best.pt'
    model = YOLOv10('best.pt')
    results = model.predict(source=pt,
                            imgsz=640)
    pre = results[0].plot()
    pre1 = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    col2.image(pre1, caption='Detected Image')
    st.balloons()
    st.success('This is a success message!', icon="✅")