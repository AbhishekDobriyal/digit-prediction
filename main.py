import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

# Create a model with the same architecture as in Colab
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compile the model (make sure to use the same compilation parameters as in Colab)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load only the model weights
model.load_weights('C:\\Users\\adobr\\PycharmProjects\\pythonProject2\\my_model_weights.h5')

def preprocess_image(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(grayscale, (28, 28))
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, [1, 28, 28])
    return reshaped_image

def predict_digit(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    return predicted_label

st.title("Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predicted_label = predict_digit(image)
        st.success(f"This Digit is recognized as {predicted_label}")

st.write("OR")

st.text("Enter the path of the image:")
image_path = st.text_input("Path:")
if st.button("Predict from Path"):
    try:
        input_image = cv2.imread(image_path)
        predicted_label = predict_digit(input_image)
        st.success(f"The Digit is recognized as {predicted_label}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
