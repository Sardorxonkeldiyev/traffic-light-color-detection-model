
![stop-light-pictures-u2lumacpdkv9srsr](https://github.com/user-attachments/assets/61d6a6b0-0689-4599-ab8d-bdee72db6fe6)

# Traffic Light Color Detection Model
This project contains a deep learning model designed to detect the color of traffic lights (Red, Green, Yellow) in real-time using a webcam. The model is trained on a dataset consisting of images categorized by traffic light colors. The repository includes the dataset, the trained model, and a Jupyter notebook demonstrating how to train and use the model.

### Project Structure
dataset/: Contains the images used to train the model. It has three subfolders:

red/: Images of red traffic lights.
green/: Images of green traffic lights.
yellow/: Images of yellow traffic lights.
traffic_light_model.h5: The pre-trained model file that can be used to detect traffic light colors.

appp.ipynb: A Jupyter notebook that includes the full training process, model architecture, and code to test the model in real-time using a webcam.

README.md: This file, which provides an overview of the project and instructions for use.

Requirements
Before running the code, make sure you have the following dependencies installed:

bash

```python
pip install tensorflow opencv-python numpy
```

### Training the Model
The model was trained using the dataset provided in the dataset folder. The dataset is split into training and validation sets, and data augmentation techniques are applied to improve the model's robustness.

### Model Architecture
The model is built using TensorFlow and Keras with the following architecture:

* Convolutional Layers: Extract features from the images.
* MaxPooling Layers: Downsample the feature maps.
* Flatten Layer: Flatten the 2D feature maps into 1D.
Dense Layers: Fully connected layers for classification.
Dropout Layer: Prevents overfitting.
Output Layer: Uses softmax activation to classify images into one of the three categories: Red, Green, Yellow.
Training Code
Here’s a snippet from the Jupyter notebook (appp.ipynb) that shows how the model was trained:


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 

# Data preparation
data_gen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = data_gen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = data_gen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model creation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(train_generator, validation_data=validation_generator, epochs=20)

# Save the model
model.save('traffic_light_model.h5')
```


* Using the Model for Real-Time Detection
You can use the trained model to detect traffic light colors in real-time using your webcam. The script below shows how to implement this:



```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('traffic_light_model.h5')

# Preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_img = preprocess_image(frame)
    prediction = model.predict(preprocessed_img)
    class_idx = np.argmax(prediction)
    color = ["Red", "Green", "Yellow"][class_idx]

    cv2.putText(frame, f'Traffic Light Color: {color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Traffic Light Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Conclusion
This project demonstrates the application of deep learning in computer vision, specifically for detecting traffic light colors in real-time. The model is trained using a simple dataset but performs well enough for basic applications. For further improvements, consider expanding the dataset or experimenting with more complex architectures.

* Acknowledgments
Thank you for checking out this project. Feel free to contribute by improving the model, adding new features, or providing feedback.
