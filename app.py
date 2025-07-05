import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer
import os

# Load the trained model
@st.cache_resource
def create_model():

    resnet_model = Sequential()

    pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                    input_shape=(224, 224, 3),
                                                    pooling='avg',
                                                    weights='imagenet')

    resnet_model.add(InputLayer(input_shape=(224, 224, 3)))

    resnet_model.add(pretrained_model)

    resnet_model.add(Flatten())

    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(BatchNormalization())

    resnet_model.add(Dense(38, activation='softmax'))

    resnet_model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    return resnet_model

# Load the model
@st.cache_resource
def load_trained_model():
    model = create_model()
    
    # Use relative path or check if file exists
    model_path = r"D:\Plant Disease Detection\model\Plant_Disease_Detection_ResNet50_model6.weights.h5"
    
    if os.path.exists(model_path):
        model.load_weights(model_path)
        return model
    else:
        st.error(f"Model weights file not found at: {model_path}")
        st.error("Please ensure the model weights file is in the same directory as this script.")
        return None

# Class mapping
classes = {
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

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Convert to RGB if image has transparency channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        img = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) 
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def format_prediction_label(label):
    """Format the prediction label for better display"""
    # Replace underscores with spaces and improve formatting
    formatted = label.replace('___', ' - ').replace('_', ' ')
    return formatted

# Streamlit App
st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect diseases using AI")

# Load model
model = load_trained_model()

if model is not None:
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...", 
        type=["jpg", "png", "jpeg"],
        help="Upload a clear image of a plant leaf for disease detection"
    )

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add prediction button
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Make prediction
                        prediction = model.predict(processed_image, verbose=0)
                        
                        # Get predicted class and confidence
                        predicted_class = np.argmax(prediction, axis=1)[0]
                        confidence = np.max(prediction) * 100
                        predicted_label = classes[predicted_class]
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Prediction", format_prediction_label(predicted_label))
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Show additional info based on prediction
                        if "healthy" in predicted_label.lower():
                            st.success("✅ The plant appears to be healthy!")
                        else:
                            st.warning("⚠️ Disease detected. Consider consulting with an agricultural expert.")
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please try uploading a different image.")

else:
    st.error("Model could not be loaded. Please check the model file path.")
    st.info("Make sure the model weights file is in the correct location.")

# Add footer
st.markdown("---")
st.markdown("*This tool is for educational purposes. For professional plant disease diagnosis, consult with agricultural experts.*")
