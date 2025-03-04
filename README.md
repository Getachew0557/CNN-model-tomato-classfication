# Tomato Disease Classification Web App

This project provides a Flask web application that classifies images of tomato leaves into various disease categories using a pre-trained deep learning model. If the image is not a tomato leaf, it is classified as "Non-tomato leaf." The classification is performed using the softmax probability distribution to identify the class with the highest probability. If the image does not belong to any known tomato disease class, it is classified as "unknown."

## Features

- **Tomato or Non-Tomato Leaf Detection**: Identifies if the input image is a tomato leaf or a non-tomato leaf before performing disease classification.
- **Image Classification**: Classifies tomato leaves into various disease categories like bacterial spots, early blight, late blight, etc.
- **Out-of-Tomato Image Detection**: Images that do not belong to any of the predefined tomato disease classes are classified as "unknown."
- **Softmax Probability**: Uses the softmax activation function to determine the class with the highest probability and checks if it meets a predefined threshold for classification. The model applies softmax activation to output a probability distribution. If the highest probability is above a predefined threshold 0.6, the image is classified into one of the known tomato disease classes. If not, it is classified as "unknown."

## Requirements

To run this application, ensure you have the following dependencies:

- Python 3.6+
- Flask
- TensorFlow
- Pillow
- NumPy

You can install the required libraries with:

```bash
pip install -r requirements.txt
