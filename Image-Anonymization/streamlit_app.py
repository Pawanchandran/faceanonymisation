import streamlit as st
import dlib
import cv2
import numpy as np

# Function to pixelate face
def pixelate_face(face_image, factor=0.05):
    h, w = face_image.shape[:2]
    kW = int(w * factor)
    kH = int(h * factor)
    if kW % 2 == 0:
        kW += 1
    if kH % 2 == 0:
        kH += 1
    return cv2.resize(cv2.GaussianBlur(face_image, (kW, kH), 0), (w, h), interpolation=cv2.INTER_NEAREST)

# Main Streamlit app
def main():
    # Title
    st.markdown("<h1 style='color:blue;text-align:left;'>Face Anonymization 🧿</h1>", unsafe_allow_html=True)

    # Description
    st.markdown("## How does it work ❓\n\nUpload any picture and it will automatically anonymize all the faces in the image.")

    # Anonymization options
    anonymization_options = ["Pixelate", "Blur", "Black Bars"]
    selected_option = st.selectbox("Select Anonymization Method", anonymization_options)

    uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        results = detector(rgb, 1)
        for r in results:
            x, y, w, h = r.rect.left(), r.rect.top(), r.rect.width(), r.rect.height()
            face_region = opencv_image[y:y + h, x:x + w]
            if selected_option == "Pixelate":
                pixelated_face = pixelate_face(face_region)
                opencv_image[y:y + h, x:x + w] = pixelated_face
            elif selected_option == "Blur":
                opencv_image[y:y + h, x:x + w] = cv2.medianBlur(face_region, 19)
            elif selected_option == "Black Bars":
                opencv_image[y:y + h, x:x + w] = [0, 0, 0]  # Black bars

        st.image(opencv_image, channels="BGR", caption="Anonymized Image", use_column_width=True)

    # Hide the Streamlit menu
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
