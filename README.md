# multi-input-cosmological-image-classifier

## Overview
This repository contains a machine learning project designed to classify cosmological maps using a neural network. The model leverages DenseNet121 for image feature extraction and combines it with cosmological parameters to improve classification accuracy. The project is implemented using TensorFlow and Keras, and includes scripts for data preprocessing, model training, evaluation, and prediction.

The main goal of this project is to advance research in astrophysics and machine learning integration by providing a comprehensive toolkit for analyzing cosmological map data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [Credits](#credits)

## Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn
- Tkinter (optional, for prediction GUI)

## Usage
### Data Preparation
Ensure you have your cosmological map images and parameters in the appropriate directories. Update the paths in the scripts if necessary.

- Image Extraction: The Image_Extraction.py script processes raw cosmological map data and extracts relevant images and parameters.
- Data Organization: Organize the extracted images and parameters into appropriate directories for training and testing.

## Data Preprocessing
The data preprocessing step involves reading cosmological map images and their corresponding parameters. The `Image_Extraction.py` script selects a subset of images, processes them, and saves the processed images along with their parameters for model training. This script ensures that the data is in the correct format for input into the neural network model.

## Model Architecture
The model architecture utilizes DenseNet121 for feature extraction from the cosmological map images. This base model is combined with additional layers that incorporate cosmological parameters, allowing the model to learn from both image data and numerical data. The architecture is defined in the `main.py` script and includes steps for data augmentation, feature extraction, and final classification.

## Training
The training process involves splitting the data into training and testing sets, performing data augmentation on the images, and training the model with early stopping to prevent overfitting. The training script also includes callbacks to monitor the training progress and save the best-performing model.

## Evaluation
After training, the model is evaluated on the test set to assess its accuracy and generalization capabilities. The `main.py` script includes functionality to plot the training history, showing both accuracy and loss over the epochs. This helps in understanding the model's performance and identifying any potential issues such as overfitting or underfitting.

## Prediction
The `predict.py` script allows for the classification of new images using the trained model. It includes a graphical user interface (GUI) for selecting images and displays the predicted category. This script demonstrates the practical application of the trained model and provides an easy-to-use interface for users.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and submit a pull request. Ensure that your contributions adhere to the project's coding standards and include appropriate tests.

## Credits
This project utilizes data from the CMD (Cosmological Maps Dataset) astrophysical dataset. Special thanks to the creators and maintainers of this dataset for providing high-quality data for research purposes.
