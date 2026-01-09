import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

st.title("🔷 Shape & Contour Analyzer")
st.write(
    "Upload an image to detect geometric shapes, count objects, "
    "and calculate area & perimeter using contour analysis."
)

# ---------------- Image Upload ----------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ---------------- Read Image ----------------
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # ---------------- Safe Grayscale Conversion ----------------
    if len(img.shape) == 3:
        if img.shape[2] == 4:   # RGBA image
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:                   # RGB image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:                       # Already grayscale
        gray = img.copy()

    # ---------------- Blur ----------------
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---------------- Threshold ----------------
    _, thresh = cv2.threshold(
        blur, 200, 255, cv2.THRESH_BINARY_INV
    )

    st.subheader("Thresholded Image")
    st.image(thresh, use_container_width=True)

    # ---------------- Find Contours ----------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ---------------- Prepare Output Image ----------------
    if len(img.shape) == 2:
        output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        output_img = img.copy()

    shape_count = {}

    # ---------------- Shape Detection ----------------
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) > 6:
            shape = "Circle"
        else:
            shape = "Polygon"

        shape_count[shape] = shape_count.get(shape, 0) + 1

        # Draw contour and label
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
        x, y = cnt[0][0]
        cv2.putText(
            output_img,
            shape,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # ---------------- Display Results ----------------
    st.subheader("Detected Shapes")
    st.image(output_img, use_container_width=True)

    st.subheader("🔢 Shape Count")
    if shape_count:
        for shape, count in shape_count.items():
            st.write(f"{shape}: {count}")
    else:
        st.write("No shapes detected.")

    st.subheader("📐 Area & Perimeter")
    obj_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 150:
            st.write(
                f"Object {obj_id}: Area = {area:.2f}, "
                f"Perimeter = {perimeter:.2f}"
            )
            obj_id += 1



