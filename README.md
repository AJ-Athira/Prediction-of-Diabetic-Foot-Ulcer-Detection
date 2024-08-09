
# Diabetic Foot Ulcer Classification using EfficientNet and InceptionResNetV2

This repository contains code for a deep learning-based image classification system designed to detect and classify Diabetic Foot Ulcers (DFUs) into various stages of severity. The model leverages pre-trained EfficientNetB3 and InceptionResNetV2 models for feature extraction and combines these features for a more robust classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project aims to classify images of diabetic foot ulcers into different stages using a deep learning model that combines the feature extraction capabilities of EfficientNetB3 and InceptionResNetV2. The model is designed to handle binary classification (presence of an ulcer) as well as multi-class classification (stage of the ulcer).

## Installation

To get started, clone this repository and install the necessary dependencies:

\`\`\`bash
git clone https://github.com/AJ-Athira/diabetic-foot-ulcer-classification.git
cd diabetic-foot-ulcer-classification
pip install -r requirements.txt
\`\`\`

## Dataset

The dataset used for this project consists of images of diabetic foot ulcers, categorized into different stages. The images should be organized in the following directory structure:

Dataset-->DFU-->Patches-->>Abnormal,Normal

- **Abnormal(Ulcer)**: Contains images of diabetic foot ulcers at various stages.
- **Normal(Healthy skin)**: Contains images of healthy skin.

Ensure that the dataset is placed in the correct directory structure as shown above.

## Model Architecture

The model architecture consists of two parallel paths for feature extraction:
- **EfficientNetB3**: Extracts features from images with a target size of 300x300 pixels.
- **InceptionResNetV2**: Extracts features from images with a target size of 299x299 pixels.

The extracted features are then concatenated and passed through dense layers for classification.

## Training

To train the model, use the following command:

\`\`\`python
python train.py
\`\`\`

Make sure to adjust the number of samples and paths in the code as per your dataset configuration.

## Evaluation

After training, the model is evaluated on a separate test set. The evaluation includes metrics such as accuracy, classification report, and confusion matrix.

## Usage

To classify a new image and predict the stage of a diabetic foot ulcer, use the \`classify_stage\` function:

\`\`\`python
result = classify_stage("path_to_image.png")
print(result)
\`\`\`

Replace \`"path_to_image.png"\` with the path to the image you want to classify.

## Results

The model provides a binary classification (Positive/Negative) as well as a multi-class classification (Stage 0 to Stage 4) of diabetic foot ulcers. The classification report and confusion matrix for the test set are displayed at the end of the training script.

## Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses pre-trained models provided by TensorFlow's Keras Applications.
- Special thanks to the creators of the EfficientNet and InceptionResNetV2 models.




