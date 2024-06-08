import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the pre-trained CNN model (EfficientNetB0)
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')


# Define a function to extract features from images
def extract_features_from_folder(folder_path):
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.endswith(('.jpg', '.jpeg', '.png'))]
    features = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(224, 224))  # Resize the image to match EfficientNet input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess the input (normalize pixel values)
        img_features = model.predict(img_array)  # Extract features using the pre-trained model
        features.append(img_features.flatten())  # Flatten the feature vector and append to the list
    return image_paths, np.array(features)


# Define a function to save features to an Excel sheet
def save_features_to_excel(image_paths, features, output_file):
    df_dict = {'Image_Path': image_paths}
    for i in range(features.shape[1]):
        df_dict[f'Feature_{i + 1}'] = features[:, i]

    # Convert dictionary to DataFrame
    df = pd.DataFrame(df_dict)

    # Check if the file exists
    if os.path.exists(output_file):
        # If the file exists, load existing data
        existing_df = pd.read_excel(output_file)
        # Append new data to existing DataFrame
        df = pd.concat([existing_df, df], ignore_index=True)

    # Save DataFrame to Excel file
    df.to_excel(output_file, index=False)


# Example usage
folder_path = r"C:\DFUDEI\datasets\DFU\Patches\Normal(Healthy skin)"  # Replace with the path to your image folder
output_file = "image_features.xlsx"
image_paths, features = extract_features_from_folder(folder_path)
save_features_to_excel(image_paths, features, output_file)
