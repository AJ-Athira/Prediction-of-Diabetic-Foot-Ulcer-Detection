import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3, InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnet_preprocess_input

# Set the random seed for reproducibility
np.random.seed(42)

# Load models once
efficientnet_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
inceptionresnet_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features using a given model and preprocessing function
def extract_features(image_paths, model, preprocess_input, target_size):
    features = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_features = model.predict(img_array)
        features.append(img_features.flatten())
    return np.array(features)

# Load images and extract features
def load_image_paths(base_dir, sub_dir, num_samples):
    image_paths = [os.path.join(base_dir, sub_dir, file) for file in os.listdir(os.path.join(base_dir, sub_dir))]
    return image_paths[:num_samples]

base_dir = "C:\\DFUDEI\\datasets\\DFU\\Patches"
num_samples = 200  # Total number of samples (adjust as needed)
abnormal_image_paths = load_image_paths(base_dir, "Abnormal(Ulcer)", num_samples // 2)
normal_image_paths = load_image_paths(base_dir, "Normal(Healthy skin)", num_samples // 2)

# Extract features using both models
abnormal_features_efficientnet = extract_features(abnormal_image_paths, efficientnet_model, efficientnet_preprocess_input, (300, 300))
normal_features_efficientnet = extract_features(normal_image_paths, efficientnet_model, efficientnet_preprocess_input, (300, 300))
abnormal_features_inceptionresnet = extract_features(abnormal_image_paths, inceptionresnet_model, inceptionresnet_preprocess_input, (299, 299))
normal_features_inceptionresnet = extract_features(normal_image_paths, inceptionresnet_model, inceptionresnet_preprocess_input, (299, 299))

# Combine features and create labels
X_abnormal_efficientnet = abnormal_features_efficientnet
X_normal_efficientnet = normal_features_efficientnet
X_abnormal_inceptionresnet = abnormal_features_inceptionresnet
X_normal_inceptionresnet = normal_features_inceptionresnet

y_abnormal = np.ones(len(abnormal_features_efficientnet))
y_normal = np.zeros(len(normal_features_efficientnet))

# Balance the classes
min_samples = min(len(X_abnormal_efficientnet), len(X_normal_efficientnet))
X_efficientnet = np.concatenate([X_abnormal_efficientnet[:min_samples], X_normal_efficientnet[:min_samples]])
X_inceptionresnet = np.concatenate([X_abnormal_inceptionresnet[:min_samples], X_normal_inceptionresnet[:min_samples]])
y = np.concatenate([y_abnormal[:min_samples], y_normal[:min_samples]])

# Split the data into training and testing sets (80-20 split)
X_train_efficientnet, X_test_efficientnet, X_train_inceptionresnet, X_test_inceptionresnet, y_train, y_test = train_test_split(
    X_efficientnet, X_inceptionresnet, y, test_size=0.2, random_state=42)

# Define the input layers
input_layer_efficientnet = Input(shape=(X_train_efficientnet.shape[1],))
input_layer_inceptionresnet = Input(shape=(X_train_inceptionresnet.shape[1],))

# Add dense layers for classification
dense_layer_efficientnet = Dense(256, activation='relu')(input_layer_efficientnet)  # Adjusted units for speed
dense_layer_inceptionresnet = Dense(256, activation='relu')(input_layer_inceptionresnet)  # Adjusted units for speed

x = Concatenate()([dense_layer_efficientnet, dense_layer_inceptionresnet])
x = Dense(256, activation='relu')(x)  # Adjusted units for speed
x = Dropout(0.5)(x)
output_layer_stage = Dense(5, activation='softmax', name='stage_output')(x)  # 5 stages: 0, 1, 2, 3, 4
output_layer_binary = Dense(1, activation='sigmoid', name='binary_output')(x)

# Create the model
model = Model(inputs=[input_layer_efficientnet, input_layer_inceptionresnet],
              outputs=[output_layer_stage, output_layer_binary])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss={'stage_output': 'sparse_categorical_crossentropy', 'binary_output': 'binary_crossentropy'},
              metrics={'stage_output': 'accuracy', 'binary_output': 'accuracy'})

# Train the model
model.fit([X_train_efficientnet, X_train_inceptionresnet], {'stage_output': y_train, 'binary_output': y_train},
          batch_size=32, epochs=50, validation_data=(
    [X_test_efficientnet, X_test_inceptionresnet], {'stage_output': y_test, 'binary_output': y_test}))

# Evaluate the model
y_pred_stage, y_pred_binary = model.predict([X_test_efficientnet, X_test_inceptionresnet])
y_pred_stage = np.argmax(y_pred_stage, axis=1)
y_pred_binary = np.round(y_pred_binary)

# Classification report
print("Binary Classification Report:")
print(classification_report(y_test, y_pred_binary))

print("Stage Classification Report:")
print(classification_report(y_test, y_pred_stage))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_stage))

# Custom classification with stage labels
def classify_stage(img_path):
    stage_names = {0: "Pre-ulcerative lesion",
                   1: "Superficial ulcer",
                   2: "Deep ulcer",
                   3: "Deep ulcer with abscess or osteomyelitis",
                   4: "Gangrene"}
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(300, 300))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Preprocess the image for EfficientNet and InceptionResNetV2
        img_array_efficientnet = efficientnet_preprocess_input(img_array.copy())
        img_array_inceptionresnet = inceptionresnet_preprocess_input(img_array)

        # Extract features
        features_efficientnet = efficientnet_model.predict(img_array_efficientnet)
        features_inceptionresnet = inceptionresnet_model.predict(img_array_inceptionresnet)

        # Predict stage and binary output
        prediction_stage, prediction_binary = model.predict([features_efficientnet, features_inceptionresnet])

        stage = np.argmax(prediction_stage)
        binary = np.round(prediction_binary)[0][0]

        if binary == 1:
            result = "Positive"
        else:
            result = "Negative"

        return f"Result: {result}\nStage: {stage}, Stage Name: {stage_names[stage]}"
    
    except Exception as e:
        print(f"Error processing image at path: {img_path}")
        print(f"Error message: {str(e)}")
        return "Error"

# Test stage classification on a sample image
sample_image_path = "C:\\DFUDEI\\normal.png"  # Replace with your image path
stage_classification = classify_stage(sample_image_path)
print("Stage Classification:", stage_classification)
