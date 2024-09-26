import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer

# Load the trained model
@st.cache_resource
def create_model():
    resnet_model = Sequential()
    pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                      input_shape=(180, 180, 3),
                                                      pooling='avg',
                                                      weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = False
    resnet_model.add(InputLayer(input_shape=(180, 180, 3)))
    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(BatchNormalization())
    resnet_model.add(Dropout(0.5))
    resnet_model.add(Dense(38, activation='softmax'))
    resnet_model.compile(optimizer=Adam(learning_rate=0.0001),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    return resnet_model

# Load the model
model = create_model()
model.load_weights("D:/Plant Disease Detection/model/Plant_Disease_Detection_ResNet50_weights.weights.h5")

# Class mapping
classes_reverse = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___healthy',
    6: 'Cherry_(including_sour)___Powdery_mildew',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___healthy',
    10: 'Corn_(maize)___Northern_Leaf_Blight',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___healthy',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___healthy',
    22: 'Potato___Late_blight',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___healthy',
    27: 'Strawberry___Leaf_scorch',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___healthy',
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
}

# Preprocess the uploaded image for prediction
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Display the interface
st.title("Plant Disease Detection")
uploaded_file = st.file_uploader("Upload a plant leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)

    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
    predicted_label = classes_reverse[predicted_class]   # Get the corresponding class label

    # Output the result
    st.write(f"Prediction: {predicted_label}")