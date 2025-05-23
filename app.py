import streamlit as st
import cv2
import numpy as np
import zipfile
import os
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Microplastic Counter", layout="wide")
st.title("🔬 Microplastic Detection Dashboard")

uploaded_zip = st.file_uploader("Upload a ZIP file of microscope images", type=["zip"])

if uploaded_zip is not None:
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall("temp_images")

    st.success("Images extracted successfully!")

    total_mp_count = 0
    image_results = []

    st.subheader("📸 Processed Images and Counts")
    for image_file in sorted(os.listdir("temp_images")):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join("temp_images", image_file)
            image = cv2.imread(img_path)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = len(contours)
            total_mp_count += count

            annotated = image.copy()
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated, caption=f"{image_file} - MPs Detected: {count}", use_column_width=True)
            image_results.append((image_file, count))

    st.markdown("---")
    st.header("📊 Summary")
    st.write(f"**Total Microplastics Counted Across All Images:** {total_mp_count}")

    st.dataframe({"Image Name": [r[0] for r in image_results], "MP Count": [r[1] for r in image_results]})

    import shutil
    shutil.rmtree("temp_images")
