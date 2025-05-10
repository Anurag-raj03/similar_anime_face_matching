import streamlit as st
import os as o
import joblib as jb
from keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import random

def save_uploaded_image(uploaded_image):
    try:
        with open(o.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False

detector = MTCNN()
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = jb.load('attribute.jbl')
anime_image = jb.load("all_imag.jbl")

def extract_feature(img_pth, model, detector):
    image = cv2.imread(img_pth)
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    x, y, width, height = results[0]['box']
    face = image[y:y+height, x:x+width]
    face_image = Image.fromarray(face)
    face_image = face_image.resize((224, 224))
    face_array = img_to_array(face_image)
    expanded_image = np.expand_dims(face_array, axis=0)
    preprocessed_image = preprocess_input(expanded_image)
    result = model.predict(preprocessed_image).flatten()
    return result

def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        sim_score = cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0]
        similarity.append(sim_score)
    return np.argmax(similarity)

st.markdown(
    """
    <style>
    .title {
        font-size: 2em;
        font-weight: bold;
        color: #33FF57;
        animation: blink 1s infinite;
        text-shadow: 2px 2px 4px #000000;
        text-align: center;
        margin-bottom: 20px;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .hover-effect:hover {
        color: #FF5733;
        text-shadow: 2px 2px 4px #000000;
    }
    .header {
        font-size: 1.5em;
        text-align: center;
    }
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .container img {
        margin: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title hover-effect">✨ Which Anime Character You Look Like !!!! ✨</div>', unsafe_allow_html=True)

uploaded_image = st.file_uploader("Choose an image")

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)

        if st.button("Submit"):
            features = extract_feature(o.path.join('uploads', uploaded_image.name), model, detector)
            if features is not None:
                index_pos = recommend(feature_list, features)
            else:
                index_pos = random.randint(0, len(anime_image) - 1)

            Character = " ".join(anime_image[index_pos].split('\\')[1].split('_'))

            st.markdown('<div class="container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.header("Your uploaded image")
                st.image(display_image, width=300)

            with col2:
                st.markdown(f'<div class="header hover-effect"><b>You Look Like {Character} 😊</b></div>', unsafe_allow_html=True)
                st.image(anime_image[index_pos], width=300)
            st.markdown('</div>', unsafe_allow_html=True)
