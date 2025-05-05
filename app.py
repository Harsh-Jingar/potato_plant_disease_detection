import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import io

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'potato_disease_detection_model.h5'
model = None

# Classes for prediction
CLASSES = ['Early Blight', 'Late Blight', 'Healthy']

# Function to load model
def load_prediction_model():
    global model
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")

# Preprocess image for prediction
def preprocess_image(img):
    # Resize image to 224x224 pixels (same as training)
    img = cv2.resize(img, (224, 224))
    # Convert to RGB if it's not already
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize the image
    img = img / 255.0
    # Convert to numpy array and add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if model is loaded
        global model
        if model is None:
            load_prediction_model()
        
        # Get the image from the POST request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        try:
            # Read and preprocess the image
            file_bytes = file.read()
            img_array = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            processed_img = preprocess_image(img)
            
            # Make prediction
            prediction = model.predict(processed_img)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = CLASSES[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index] * 100)
            
            # Return prediction result
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'all_probabilities': {
                    class_name: float(pred * 100) 
                    for class_name, pred in zip(CLASSES, prediction[0])
                }
            })
        
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Load model at startup
    load_prediction_model()
    app.run(debug=True)