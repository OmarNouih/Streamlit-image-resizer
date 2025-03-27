import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import requests
from urllib.request import urlopen
import os

# Function for resizing and displaying the image
def detect_and_resize_content(img, expand_factor=1.1, min_target_size=920, canvas_size=1000):
    original_height, original_width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
        return img_pil, "🚀 No main content detected. Keeping the original image.", False

    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    width_ratio = w / original_width
    height_ratio = h / original_height

    if width_ratio >= 0.88 and height_ratio >= 0.88:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
        return img_pil, "✅ Image is already well-centered and well-sized. No changes applied.", False

    expand_x = max(20, int((expand_factor - 1) * w / 2))
    expand_y = max(20, int((expand_factor - 1) * h / 2))
    x = max(0, x - expand_x)
    y = max(0, y - expand_y)
    w = min(original_width - x, w + 2 * expand_x)
    h = min(original_height - y, h + 2 * expand_y)

    img_cropped = img[y:y + h, x:x + w]
    img_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGBA))

    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = min_target_size
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = min_target_size
        new_w = int(new_h * aspect_ratio)

    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    background = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2
    background.paste(img_pil, (x_offset, y_offset), img_pil)

    resize_ratio_w = new_w / w
    resize_ratio_h = new_h / h
    max_resize_ratio = max(resize_ratio_w, resize_ratio_h)

    if max_resize_ratio < 1.1:
        return background, "➖ Minor resize only (<10%), treated as unchanged.", False
    if max_resize_ratio > 1.3:
        return background, "⚠️ Warning: Image might have been resized incorrectly due to large expansion.", False

    return background, "✅ Image properly resized and centered.", True

# Function to load image from URL
def load_image_from_url(url):
    try:
        img_array = np.array(bytearray(urlopen(url).read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)  # Decode to OpenCV format
        return img
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# Streamlit App
def main():
    st.title("Image Resizer App")

    # Option to upload image or input URL
    option = st.selectbox("Choose Image Input Method", ["Upload Image", "Provide Image URL"])
    
    img = None
    image_name = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert to OpenCV format for processing
            img = np.array(image)
            image_name = uploaded_file.name  # Save the original file name for download

    elif option == "Provide Image URL":
        image_url = st.text_input("Enter Image URL")
        if image_url:
            # Load image from URL
            img = load_image_from_url(image_url)
            if img is not None:
                # Display the image
                st.image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), caption="Image from URL", use_container_width=True)
                image_name = os.path.basename(image_url)  # Extract file name from URL

    if img is not None:
        # Apply resizing based on your logic
        resized_image, status, is_modified = detect_and_resize_content(img)

        # Afficher l’image centrée
        st.image(resized_image, caption="Resized Image", use_container_width=True)

        # Centrer la notification (statut)
        st.markdown(
            f"<div style='text-align: center; font-size: 18px; margin: 15px 0;'>{status}</div>",
            unsafe_allow_html=True
        )

        # Télécharger le résultat
        buffer = BytesIO()
        resized_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Centrer le bouton de téléchargement
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Télécharger l'image redimensionnée",
            data=buffer,
            file_name=image_name if image_name else "resized_image.png",
            mime="image/png"
        )
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
