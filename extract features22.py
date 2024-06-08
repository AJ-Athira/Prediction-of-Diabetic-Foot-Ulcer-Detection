import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

# Set the random seed for reproducibility
np.random.seed(42)

# Function to extract features using EfficientNetB3
def extract_efficientnet_features(image_paths):
    model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(300, 300))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = efficientnet_preprocess_input(img_array)
        img_features = model.predict(img_array)
        features.append(img_features.flatten())
    return np.array(features)

# Load images and extract features
abnormal_image_paths = [os.path.join("C:\\DFUDEI\\datasets\\DFU\\Patches\\Abnormal(Ulcer)", file) for file in os.listdir("C:\\DFUDEI\\datasets\\DFU\\Patches\\Abnormal(Ulcer)")]
normal_image_paths = [os.path.join("C:\\DFUDEI\\datasets\\DFU\\Patches\\Normal(Healthy skin)", file) for file in os.listdir("C:\\DFUDEI\\datasets\\DFU\\Patches\\Normal(Healthy skin)")]

abnormal_features = extract_efficientnet_features(abnormal_image_paths)
normal_features = extract_efficientnet_features(normal_image_paths)

# Combine features and create labels
X = np.concatenate([abnormal_features, normal_features])
y = np.concatenate([np.ones(len(abnormal_features)), np.zeros(len(normal_features))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the input layer
input_layer = Input(shape=(X_train.shape[1],))

# Add the classification head
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred)

# Classification report
print(classification_report(y_test, y_pred_binary))

# Custom classification
def classify_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(300, 300))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = efficientnet_preprocess_input(img_array)
        img_features = extract_efficientnet_features([img_path])
        prediction = model.predict(img_features)
        if prediction >= 0.5:
            return "Positive"
        else:
            return "Negative"
    except Exception as e:
        print(f"Error processing image at path: {img_path}")
        print(f"Error message: {str(e)}")
        return "Error"

# Test classification on a sample image
sample_image_path = "C:\\DFUDEI\\normal.png"
classification = classify_image(sample_image_path)
print("Classification:", classification)
