import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Function to predict the image class
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1], decoded_predictions[0][0][2]  # Return class name and probability

# Test with an image
image_path = r"C:\Users\Harish\Downloads\human.jpg"  
print(os.path.exists(image_path))  # Should print True if file exists
class_name, probability = predict_image(image_path)
print(f"Predicted class: {class_name} with probability: {probability:.2f}")

# Display the image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(f'{class_name} ({probability:.2f})')
plt.show()
