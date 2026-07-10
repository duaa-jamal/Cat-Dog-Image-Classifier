# ============================================================
# STEP 3: PREDICT - Classify a New Image
# Run this AFTER training is complete
# ============================================================

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os
import sys

print("=" * 60)
print("STEP 3: Animal Classifier - Prediction")
print("=" * 60)

# -------------------------------------------------------
# CONFIGURATION - CHANGE IMAGE PATH HERE
# -------------------------------------------------------
IMAGE_PATH = "C:/Users/Noman Traders/Downloads/files/dataset/test_set/cats/cat.4973.jpg"  # ← Change this to your image path
                            # Example: "C:/Users/YourName/Desktop/cat.jpg"

# -------------------------------------------------------
# LOAD CLASS LABELS
# -------------------------------------------------------
if not os.path.exists("class_labels.json"):
    print(" class_labels.json not found!")
    print("Please run Step2_train_model.py first")
    exit()

with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

print(f" Loaded class labels: {class_indices}")

# Build reverse mapping: {0: 'cats', 1: 'dogs'}
# This is KEY - we flip the dictionary to go from number → name
index_to_class = {v: k for k, v in class_indices.items()}
print(f" Reverse mapping: {index_to_class}")

# -------------------------------------------------------
# CHECK IMAGE EXISTS
# -------------------------------------------------------
if not os.path.exists(IMAGE_PATH):
    print(f"\n ERROR: Image not found at: {IMAGE_PATH}")
    print(f"Please make sure the image exists at that path!")
    exit()

print(f"\n  Loading image: {IMAGE_PATH}")

# -------------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------------
# Use the BEST model (saved by ModelCheckpoint)
model_path = "animal_model_best.h5"
if not os.path.exists(model_path):
    model_path = "animal_model.h5"   # fallback to regular model

if not os.path.exists(model_path):
    print(" No trained model found!")
    print("Please run Step2_train_model.py first")
    exit()

print(f" Loading model: {model_path}")
model = load_model(model_path)
print(" Model loaded!")

# -------------------------------------------------------
# PREPROCESS IMAGE
# -------------------------------------------------------
# Load image and resize to same size used during training
test_image = image.load_img(IMAGE_PATH, target_size=(64, 64))

# Convert image to numpy array (shape: 64, 64, 3)
test_image = image.img_to_array(test_image)

# Normalize pixel values from [0-255] to [0-1]
# VERY IMPORTANT: must match training preprocessing
test_image = test_image / 255.0

# Add batch dimension (shape: 1, 64, 64, 3)
# Model expects batches, so we add a dimension
test_image = np.expand_dims(test_image, axis=0)

print(f" Image preprocessed, shape: {test_image.shape}")

# -------------------------------------------------------
# MAKE PREDICTION
# -------------------------------------------------------
print("\n Making prediction...")
result = model.predict(test_image, verbose=0)
prediction_value = float(result[0][0])

print(f"\n Raw prediction value: {prediction_value:.4f}")
print("   (0.0 = definitely cats, 1.0 = definitely dogs)")

# -------------------------------------------------------
# INTERPRET RESULT
# -------------------------------------------------------
# The model outputs a value between 0 and 1
# We need to figure out which class it belongs to

if prediction_value > 0.5:
    predicted_class_index = 1
else:
    predicted_class_index = 0

# Get the class name from our reverse mapping
predicted_animal = index_to_class[predicted_class_index].upper()

# Calculate confidence percentage
if prediction_value > 0.5:
    dog_confidence = prediction_value * 100
    cat_confidence = (1 - prediction_value) * 100
else:
    cat_confidence = (1 - prediction_value) * 100
    dog_confidence = prediction_value * 100

# -------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print(" PREDICTION RESULT")
print("=" * 60)
print(f"🐾 Detected Animal: {predicted_animal}")
print(f"🐱 Cat Probability: {cat_confidence:.1f}%")
print(f"🐶 Dog Probability: {dog_confidence:.1f}%")

# Confidence level interpretation
max_confidence = max(cat_confidence, dog_confidence)
if max_confidence > 90:
    confidence_label = "Very High Confidence "
elif max_confidence > 75:
    confidence_label = "High Confidence "
elif max_confidence > 60:
    confidence_label = "Medium Confidence "
else:
    confidence_label = "Low Confidence  (try a clearer image)"

print(f" Confidence Level: {confidence_label}")
print("=" * 60)

# -------------------------------------------------------
# SHOW IMAGE WITH RESULT (Optional visual)
# -------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(IMAGE_PATH)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(
        f"Predicted: {predicted_animal}\n"
        f"Cat: {cat_confidence:.1f}% | Dog: {dog_confidence:.1f}%",
        fontsize=14,
        fontweight='bold',
        color='green' if max_confidence > 75 else 'orange'
    )
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    print(" Result image saved as prediction_result.png")
except Exception as e:
    print(f"(Could not display image: {e})")
