# Cat & Dog Image Classifier

A deep learning project that classifies images as either a cat or a dog using a Convolutional Neural Network (CNN). The model is trained on a labeled image dataset and can predict the class of new images with confidence scores. The project also includes a desktop application for easy image classification.

## Features

- Classifies images as cats or dogs
- CNN model built with TensorFlow and Keras
- Train the model on a custom dataset
- Predict a single image
- Desktop GUI for image classification
- Model evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- Saves the trained model and training graphs

## Project Files

- `Step1_download_dataset.py` – Creates the required dataset folders.
- `Step2_train_model.py` – Trains the CNN model.
- `Step3_predict.py` – Predicts whether an image is a cat or a dog.
- `Step4_evaluate_model.py` – Evaluates the trained model.
- `Step5_desktop_app.py` – Launches the desktop application.

## Installation

Install the required libraries:

```bash
pip install tensorflow keras numpy matplotlib pillow scikit-learn seaborn
```

## Dataset

This project uses the Cat and Dog dataset from Kaggle.

https://www.kaggle.com/datasets/tongpython/cat-and-dog

Download the dataset and run:

```bash
python Step1_download_dataset.py
```

Then copy the downloaded images into the generated folders.

## Training

Run:

```bash
python Step2_train_model.py
```

This will generate:

- `animal_model.h5`
- `animal_model_best.h5`
- `class_labels.json`
- `training_results.png`

## Prediction

Run:

```bash
python Step3_predict.py
```

Update the `IMAGE_PATH` variable with the image you want to classify before running the script.

Example output:

```text
Prediction Result

Detected Animal: Cat

Cat Probability: 87.3%
Dog Probability: 12.7%

Confidence: High
```

## Evaluation

Run:

```bash
python Step4_evaluate_model.py
```

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Desktop Application

Run:

```bash
python Step5_desktop_app.py
```

The desktop application allows users to select an image and receive a prediction through a simple graphical interface.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## Future Improvements

- Improve accuracy using transfer learning
- Support more animal classes
- Deploy as a web application
- Add real-time webcam detection

## Author

Duaa Jamal

GitHub: https://github.com/duaa-jamal
