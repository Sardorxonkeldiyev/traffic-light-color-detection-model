
![resized_image](https://github.com/user-attachments/assets/0dbecf02-cc1c-454d-a376-0faec530fe88)

# 🚦 Traffic Light Color Detection (Computer Vision)

## 📝 Project Overview
This repository contains a **Deep Learning** and **Computer Vision** project designed to detect and classify traffic light colors (**Red, Green, Yellow**). Built using TensorFlow and Keras, the project features a custom Convolutional Neural Network (CNN) and a live detection script that utilizes the computer's webcam to classify traffic lights in real-time.

## 🛠️ Tools & Technologies
* **Deep Learning Framework:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation:** NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Model Evaluation:** Scikit-Learn (Confusion Matrix, Classification Report)

## 🚀 Key Features & Workflow

### 1. Image Data Preprocessing & Augmentation
Used `ImageDataGenerator` to normalize pixel values and artificially expand the training dataset. Applied random transformations (rotation, shifting, zooming, and flipping) to make the model more robust and prevent overfitting.

### 2. CNN Architecture
Built a custom Sequential Convolutional Neural Network consisting of:
* **3 Convolutional Blocks:** `Conv2D` + `MaxPooling2D` with ReLU activation to extract spatial features from images.
* **Fully Connected Layers:** `Flatten` layer followed by `Dense` layers.
* **Regularization:** `Dropout (0.5)` layer included to mitigate overfitting.
* **Output Layer:** `Softmax` activation predicting 3 specific classes (Red, Green, Yellow).

### 3. Model Evaluation
* Visualized Training vs. Validation **Accuracy and Loss** curves to monitor the learning process.
* Generated a **Confusion Matrix** (via Seaborn heatmap) and **Classification Report** to thoroughly analyze precision, recall, and f1-scores for each specific color class.

### 4. Real-Time Detection (Live Webcam)
Loaded the trained `traffic_light_model_advanced.h5` model into a live testing environment. Using OpenCV, the script captures webcam frames, preprocesses them on the fly (resizing to 128x128 and normalizing), and overlays the real-time prediction text directly onto the video stream.

## 📂 Project Structure
* `dataset/`: Directory containing the training and validation images (must be organized into 'Red', 'Green', and 'Yellow' subfolders).
* `traffic_light_model_advanced.h5`: The saved trained deep learning model.
* `main_script.py` / `notebook.ipynb`: Contains the model training, evaluation, and live webcam detection code.

---
*Created by Sardor - Aspiring Data Professional (Computer Vision & Data Science)*
