# Potato Plant Disease Detection

![Potato Plant Disease Detection](https://img.shields.io/badge/AI%20Project-Plant%20Disease%20Detection-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-blue)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)

An AI-powered web application that detects diseases in potato plants from leaf images using deep learning.

## Project Overview

This project uses a Convolutional Neural Network (CNN) to classify potato plant leaves into three categories:
- Early Blight (fungal disease caused by *Alternaria solani*)
- Late Blight (water mold disease caused by *Phytophthora infestans*)
- Healthy

The model was trained on a dataset of thousands of potato leaf images and provides not only classification results but also confidence scores and treatment recommendations for detected diseases.

## Features

- **Real-time Disease Detection**: Upload an image and get immediate diagnosis
- **User-friendly Interface**: Clean, responsive design with drag-and-drop functionality
- **Detailed Results**: Provides disease information, confidence scores, and treatment recommendations
- **Pre-trained Model**: Uses a trained deep learning model for accurate predictions

## Technologies Used

- **TensorFlow/Keras**: For building and training the deep learning model
- **OpenCV**: For image preprocessing
- **Flask**: For the web server backend
- **HTML/CSS/JavaScript**: For the frontend user interface
- **Bootstrap**: For responsive design components

## Dataset

The model was trained on the Potato Disease Dataset, which contains thousands of labeled images of potato plant leaves across three classes:
- Early Blight: ~1,000 images
- Late Blight: ~1,000 images  
- Healthy: ~1,000 images

The dataset is organized in the standard format for image classification tasks, with separate folders for each class.

## Model Architecture

The disease detection model is a deep convolutional neural network with the following characteristics:
- Input shape: 224x224x3 (RGB images)
- Multiple convolutional and pooling layers
- Dropout layers to prevent overfitting
- Dense layers with ReLU activation
- Softmax output layer for multi-class classification

The model achieved:
- Accuracy: ~96% on validation data
- Loss: ~0.14 on validation data

## Training Result
![image](https://github.com/user-attachments/assets/bd7e9846-b933-4c65-9f02-0da1168fe85e)

![image](https://github.com/user-attachments/assets/3b72ca55-d259-4e6d-92db-8c89d6823524)

![image](https://github.com/user-attachments/assets/5c07fd33-96a7-4d1e-bb70-f980bb9b211e)

![Screenshot 2025-05-05 143702](https://github.com/user-attachments/assets/03313814-b4a0-42f4-a2a0-338665d54f9b)

## Installation and Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/potato-disease-detection.git
cd potato-disease-detection
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Run the application:
```
python app.py
```

4. Open a web browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. Launch the application in your browser
2. Upload an image of a potato plant leaf by:
   - Clicking the "Choose File" button
   - Dragging and dropping an image onto the upload area
3. Click "Analyze Image" to process the image
4. View the results, including:
   - Disease classification
   - Confidence percentage
   - Disease information and treatment recommendations

## Project Structure

```
potato-disease-detection/
├── app.py                          # Flask web application
├── potato_disease_detection_model.h5   # Trained model file
├── best_potato_disease_model.h5    # Best model checkpoint
├── potato_disease_model_training.ipynb  # Model training notebook
├── requirements.txt                # Python dependencies
├── datasets/                       # Dataset directory
│   ├── Potato___Early_blight/      # Early blight images
│   ├── Potato___Late_blight/       # Late blight images
│   └── Potato___healthy/           # Healthy plant images
└── templates/                      # HTML templates
    └── index.html                  # Main webpage
```

## Future Improvements

- Add support for more plant types and diseases
- Implement mobile app version for field use
- Add image enhancement options for better predictions in suboptimal lighting
- Integrate with a larger plant health monitoring system
- Add geolocation data to track disease spread

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used for training is based on publicly available plant disease datasets
- Inspired by research in computer vision applications for sustainable agriculture

---

*This project was developed as part of an AI/ML internship to showcase deep learning applications in agricultural technology.*
