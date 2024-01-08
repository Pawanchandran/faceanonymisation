from helpers import convert_and_trim_bb
import streamlit as st
import imutils
import dlib
import cv2
import numpy as np

# Function to pixelate face
# Function to pixelate face
def pixelate_face(face_image, factor=0.05):
    # Get the dimensions of the face image
    height, width, _ = face_image.shape

    # Calculate the number of pixels to represent each dimension based on the factor
    h_pix = int(height * factor)
    w_pix = int(width * factor)

    # Resize the face image down to a small size and then up to pixelate it
    small = cv2.resize(face_image, (w_pix, h_pix), interpolation=cv2.INTER_LINEAR)
    pixelated_face = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    return pixelated_face


# Main Streamlit app
def main():
    # Title
    st.markdown("<h1 style='color:blue;text-align:left;'>Image Anonymization 🧿</h1>", unsafe_allow_html=True)

    # Description
    st.markdown("## How does it work ❓\n\nUpload any picture and it will automatically anonymize all the faces in the image.")

    # File uploader
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
            if st.checkbox("Pixelate"):
                pixelated_face = pixelate_face(face_region)
                opencv_image[y:y + h, x:x + w] = pixelated_face
            elif st.checkbox("Blur"):
                opencv_image[y:y + h, x:x + w] = cv2.medianBlur(face_region, 19)
            elif st.checkbox("Black Bars"):
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
