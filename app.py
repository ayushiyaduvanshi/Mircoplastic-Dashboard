import streamlit as st
import cv2
import numpy as np
import zipfile
import os
import io
import pandas as pd
from PIL import Image

# =========================
# Utility functions
# =========================
def classify_shape(cnt, eps_frac=0.01):
    """Return (aspect_ratio, circularity, solidity, extent, label) for a contour."""
    area = cv2.contourArea(cnt)
    if area <= 0:
        return 0, 0, 0, 0, "noise"

    # Perimeter
    peri = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    # Bounding rect metrics
    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h) if (w * h) > 0 else 0
    aspect_ratio = max(w, h) / max(1, min(w, h))

    # Convex hull solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Simple rule-based labeling (tweak thresholds below)
    label = "fragment"
    if aspect_ratio >= st.session_state["fiber_ar_min"] and solidity > st.session_state["fiber_sol_min"] and circularity < 0.8:
        label = "fiber"
    elif circularity >= st.session_state["bead_circ_min"] and solidity > st.session_state["bead_sol_min"]:
        label = "bead"
    elif (2.0 <= aspect_ratio < st.session_state["fiber_ar_min"]) and (solidity < st.session_state["film_sol_max"]) and (extent < st.session_state["film_extent_max"]):
        label = "film"
    else:
        label = "fragment"

    return aspect_ratio, circularity, solidity, extent, label


def equivalent_diameter_um(area_px, um_per_px):
    # Eq. diameter from area: d = sqrt(4A/π)
    if area_px <= 0:
        return 0.0
    d_px = np.sqrt(4 * area_px / np.pi)
    return d_px * um_per_px


def color_mask_hsv(image_bgr, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
    return mask


def preprocess_and_find_contours(image_bgr, use_color, hsv_lower, hsv_upper, blur_ksize, morph_ksize, canny_low, canny_high, min_area_px):
    if use_color:
        mask = color_mask_hsv(image_bgr, hsv_lower, hsv_upper)
        # Morphology to clean noise
        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if blur_ksize > 0:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        edges = cv2.Canny(gray, canny_low, canny_high)
        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area in px
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    return contours


def draw_annotations(image_bgr, contours, meta_list, thickness=2):
    annotated = image_bgr.copy()
    for cnt, meta in zip(contours, meta_list):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        label = f"{meta['shape']} | {meta['eq_d_um']:.0f}µm"
        cv2.putText(annotated, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def bin_size_um(d_um, bins_um):
    """Return a bin label for the given diameter d_um and sorted bins e.g. [300, 1000]."""
    if d_um <= bins_um[0]:
        return f"≤{int(bins_um[0])}µm"
    elif d_um <= bins_um[1]:
        return f"{int(bins_um[0])}-{int(bins_um[1])}µm"
    else:
        return f">{int(bins_um[1])}µm"


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Microplastic Counter (Size + Shape)", layout="wide")
st.title("🔬 Microplastic Detection Dashboard — Size & Shape")

with st.sidebar:
    st.header("Calibration & Filters")

    um_per_px = st.number_input("Microns per pixel (calibration)", value=1.50, min_value=0.0001, step=0.01, help="Set from your microscope scale bar.")
    min_d_um = st.number_input("Minimum equivalent diameter to count (µm)", value=50.0, min_value=0.0, step=10.0)
    max_d_um = st.number_input("Maximum equivalent diameter to count (µm)", value=10000.0, min_value=0.0, step=50.0)

    st.markdown("---")
    st.subheader("Detection Mode")
    use_color = st.checkbox("Use HSV color mask (otherwise edge-based)", value=True)

    if use_color:
        st.caption("HSV ranges (H:0-179, S:0-255, V:0-255). Add blue/green/etc. as needed.")
        h1 = st.slider("H lower", 0, 179, 90)
        s1 = st.slider("S lower", 0, 255, 50)
        v1 = st.slider("V lower", 0, 255, 50)
        h2 = st.slider("H upper", 0, 179, 130)
        s2 = st.slider("S upper", 0, 255, 255)
        v2 = st.slider("V upper", 0, 255, 255)
        hsv_lower = (h1, s1, v1)
        hsv_upper = (h2, s2, v2)
    else:
        canny_low = st.slider("Canny low threshold", 0, 255, 50)
        canny_high = st.slider("Canny high threshold", 0, 255, 150)

    blur_ksize = st.slider("Gaussian blur kernel (odd, 0 disables)", 0, 25, 5, step=1)
    if blur_ksize % 2 == 0 and blur_ksize != 0:
        blur_ksize += 1  # ensure odd
    morph_ksize = st.slider("Morphology kernel", 1, 15, 3, step=1)

    st.markdown("---")
    st.subheader("Shape Rules (tune if needed)")
    # default thresholds into session_state for classify_shape
    if "fiber_ar_min" not in st.session_state:
        st.session_state["fiber_ar_min"] = 5.0
        st.session_state["fiber_sol_min"] = 0.2
        st.session_state["bead_circ_min"] = 0.8
        st.session_state["bead_sol_min"] = 0.9
        st.session_state["film_sol_max"] = 0.9
        st.session_state["film_extent_max"] = 0.65

    st.session_state["fiber_ar_min"] = st.slider("Fiber: min aspect ratio", 2.0, 15.0, st.session_state["fiber_ar_min"], 0.5)
    st.session_state["fiber_sol_min"] = st.slider("Fiber: min solidity", 0.0, 1.0, st.session_state["fiber_sol_min"], 0.05)
    st.session_state["bead_circ_min"] = st.slider("Bead: min circularity", 0.0, 1.0, st.session_state["bead_circ_min"], 0.05)
    st.session_state["bead_sol_min"] = st.slider("Bead: min solidity", 0.0, 1.0, st.session_state["bead_sol_min"], 0.05)
    st.session_state["film_sol_max"] = st.slider("Film: max solidity", 0.2, 1.0, st.session_state["film_sol_max"], 0.05)
    st.session_state["film_extent_max"] = st.slider("Film: max extent", 0.2, 1.0, st.session_state["film_extent_max"], 0.05)

    st.markdown("---")
    st.subheader("Size Bins (for summary)")
    bin1 = st.number_input("Bin 1 upper (µm)", value=300.0, min_value=1.0, step=10.0)
    bin2 = st.number_input("Bin 2 upper (µm)", value=1000.0, min_value=bin1 + 1.0, step=10.0)
    bins_um = [bin1, bin2]

uploaded_zip = st.file_uploader("Upload a ZIP file of microscope images", type=["zip"])

if uploaded_zip is not None:
    # Extract images
    extract_dir = "temp_images"
    if os.path.exists(extract_dir):
        # clean previous run
        for f in os.listdir(extract_dir):
            try:
                os.remove(os.path.join(extract_dir, f))
            except Exception:
                pass
    else:
        os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    st.success("Images extracted successfully!")

    # Per-image results and global summary
    per_image_rows = []
    per_particle_rows = []

    st.subheader("📸 Processed Images with Annotations")
    img_files = [f for f in sorted(os.listdir(extract_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))]

    total_count = 0
    shape_tally = {"fiber": 0, "fragment": 0, "bead": 0, "film": 0}
    size_bin_tally = {f"≤{int(bins_um[0])}µm": 0, f"{int(bins_um[0])}-{int(bins_um[1])}µm": 0, f">{int(bins_um[1])}µm": 0}

    for image_file in img_files:
        img_path = os.path.join(extract_dir, image_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Preprocess & find contours
        if use_color:
            contours = preprocess_and_find_contours(
                image, True, hsv_lower, hsv_upper, blur_ksize, morph_ksize, None, None, 1
            )
        else:
            contours = preprocess_and_find_contours(
                image, False, None, None, blur_ksize, morph_ksize, canny_low, canny_high, 1
            )

        # Build particle metadata & apply size filtering in µm
        particles_meta = []
        img_count = 0
        img_shape_tally = {"fiber": 0, "fragment": 0, "bead": 0, "film": 0}

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            d_um = equivalent_diameter_um(area_px, um_per_px)
            if not (min_d_um <= d_um <= max_d_um):
                continue

            ar, circ, sol, ext, label = classify_shape(cnt)

            particles_meta.append({
                "eq_d_um": d_um,
                "area_px": area_px,
                "aspect_ratio": ar,
                "circularity": circ,
                "solidity": sol,
                "extent": ext,
                "shape": label
            })

            img_count += 1
            shape_tally[label] += 1
            img_shape_tally[label] = img_shape_tally.get(label, 0) + 1

            size_label = bin_size_um(d_um, bins_um)
            size_bin_tally[size_label] += 1

            # Particle-level row for CSV
            per_particle_rows.append({
                "image": image_file,
                "eq_d_um": round(d_um, 2),
                "aspect_ratio": round(ar, 3),
                "circularity": round(circ, 3),
                "solidity": round(sol, 3),
                "extent": round(ext, 3),
                "shape": label
            })

        total_count += img_count

        # Annotate & show
        annotated = draw_annotations(image, [c for c in contours if cv2.contourArea(c) > 0][:len(particles_meta)], particles_meta)
        st.image(annotated, caption=f"{image_file} — MPs: {img_count} | {img_shape_tally}", use_column_width=True)

        # Per-image summary row
        per_image_rows.append({
            "Image Name": image_file,
            "Total MPs": img_count,
            "Fibers": img_shape_tally.get("fiber", 0),
            "Fragments": img_shape_tally.get("fragment", 0),
            "Beads": img_shape_tally.get("bead", 0),
            "Films": img_shape_tally.get("film", 0)
        })

    st.markdown("---")
    st.header("📊 Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total MPs Counted (all images)", value=total_count)
        st.write("**By Shape**")
        st.dataframe(pd.DataFrame.from_dict(shape_tally, orient="index", columns=["Count"]))
    with col2:
        st.write("**By Size Bin (Equivalent Diameter)**")
        st.dataframe(pd.DataFrame.from_dict(size_bin_tally, orient="index", columns=["Count"]))

    # Tables
    per_image_df = pd.DataFrame(per_image_rows).sort_values("Image Name")
    st.subheader("Per-Image Summary")
    st.dataframe(per_image_df, use_container_width=True)

    per_particle_df = pd.DataFrame(per_particle_rows)
    if not per_particle_df.empty:
        st.subheader("Per-Particle Measurements")
        st.dataframe(per_particle_df, use_container_width=True)

        # CSV downloads
        buf_img = io.BytesIO()
        per_image_df.to_csv(buf_img, index=False)
        st.download_button("Download Per-Image Summary (CSV)", data=buf_img.getvalue(),
                           file_name="mp_per_image_summary.csv", mime="text/csv")

        buf_part = io.BytesIO()
        per_particle_df.to_csv(buf_part, index=False)
        st.download_button("Download Per-Particle Measurements (CSV)", data=buf_part.getvalue(),
                           file_name="mp_per_particle_measurements.csv", mime="text/csv")

    # Cleanup extracted images (optional; comment out if you want to inspect files)
    try:
        for f in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, f))
        os.rmdir(extract_dir)
    except Exception:
        pass

else:
    st.info("Upload a ZIP of microscope images to begin. Tip: set microns-per-pixel from your scale bar for accurate size.")
