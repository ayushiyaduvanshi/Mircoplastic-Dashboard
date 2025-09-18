import streamlit as st
import cv2
import numpy as np
import zipfile
import os
import io
import pandas as pd

# -------------------------
# Simple shape classifier
# -------------------------
def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    if area <= 0:
        return "noise", 0, 0, 0, 0

    peri = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = (max(w, h) / max(1, min(w, h)))
    extent = (area / (w * h)) if (w * h) > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = (area / hull_area) if hull_area > 0 else 0

    # Thresholds (tuned for simplicity; adjust if needed)
    if aspect_ratio >= 5.0 and solidity > 0.2 and circularity < 0.8:
        label = "fiber"
    elif circularity >= 0.8 and solidity > 0.9:
        label = "bead"
    elif (2.0 <= aspect_ratio < 5.0) and (solidity < 0.9) and (extent < 0.65):
        label = "film"
    else:
        label = "fragment"

    return label, aspect_ratio, circularity, solidity, extent


def eq_diameter_um(area_px, um_per_px):
    if area_px <= 0:
        return 0.0
    d_px = np.sqrt(4 * area_px / np.pi)
    return float(d_px * um_per_px)


def detect_blue_mask(image_bgr):
    """Return binary mask for 'blue' in HSV."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Broad blue range (tweak if needed)
    lower_blue = np.array([90, 50, 50], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def annotate(image_bgr, contours, meta_list):
    out = image_bgr.copy()
    for cnt, meta in zip(contours, meta_list):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{meta['shape']} | {int(round(meta['size_um']))}µm"
        cv2.putText(out, label, (x, max(15, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# -------------------------
# UI (no sidebar)
# -------------------------
st.set_page_config(page_title="Blue Microplastic Detector — Size & Shape", layout="wide")
st.title("🔵 Microplastic Detection (Blue) — Size & Shape")

um_per_px = st.number_input(
    "Microns per pixel (calibration from your microscope scale bar)",
    min_value=0.0001, value=1.50, step=0.01, help="Example: if 100 µm = 80 px, then µm/px = 100/80 = 1.25"
)

uploaded_zip = st.file_uploader("Upload a ZIP of microscope images (.jpg/.jpeg/.png)", type=["zip"])

if uploaded_zip is not None:
    # Prepare temp dir
    extract_dir = "temp_images"
    os.makedirs(extract_dir, exist_ok=True)
    for f in os.listdir(extract_dir):
        try:
            os.remove(os.path.join(extract_dir, f))
        except:
            pass

    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        zf.extractall(extract_dir)

    st.success("Images extracted. Processing…")

    img_files = [f for f in sorted(os.listdir(extract_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    per_particle_rows = []
    total_count = 0

    st.subheader("Annotated Images")
    for image_file in img_files:
        p = os.path.join(extract_dir, image_file)
        img = cv2.imread(p)
        if img is None:
            continue

        mask = detect_blue_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        meta_list = []
        for idx, cnt in enumerate(contours, start=1):
            area_px = cv2.contourArea(cnt)
            if area_px <= 0:
                continue

            size_um = eq_diameter_um(area_px, um_per_px)
            shape, ar, circ, sol, ext = classify_shape(cnt)

            per_particle_rows.append({
                "image": image_file,
                "particle_id": idx,
                "shape": shape,
                "size_um": round(size_um, 2),
                "aspect_ratio": round(ar, 3),
                "circularity": round(circ, 3),
                "solidity": round(sol, 3),
                "extent": round(ext, 3),
                "area_px": int(round(area_px))
            })
            meta_list.append({"shape": shape, "size_um": size_um})

        total_count += len(meta_list)
        annotated = annotate(img, contours, meta_list)
        st.image(annotated, caption=f"{image_file} — blue MPs: {len(meta_list)}", use_column_width=True)

    # -------------------------
    # Final Summary (bottom)
    # -------------------------
    st.markdown("---")
    st.header("📊 Summary (All Images)")

    st.metric("Total blue particles detected", total_count)

    if per_particle_rows:
        df = pd.DataFrame(per_particle_rows)
        # Shape distribution
        shape_counts = df["shape"].value_counts().reset_index()
        shape_counts.columns = ["shape", "count"]

        st.subheader("Per-Particle Details")
        st.dataframe(df, use_container_width=True)

        st.subheader("Shape Distribution")
        st.dataframe(shape_counts, use_container_width=True)

        # CSV download
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Per-Particle Summary (CSV)",
            data=csv_buf.getvalue(),
            file_name="blue_mp_per_particle_summary.csv",
            mime="text/csv"
        )
    else:
        st.info("No blue particles detected in the uploaded images.")

    # Cleanup (optional)
    try:
        for f in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, f))
        os.rmdir(extract_dir)
    except:
        pass
else:
    st.info("Upload a ZIP to run detection. Enter the correct µm/px calibration for accurate sizes.")
