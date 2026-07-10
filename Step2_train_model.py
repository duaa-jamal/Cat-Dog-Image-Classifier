# ============================================================
# STEP 2: TRAIN THE MODEL
# Run this AFTER placing images in the dataset folders
# ============================================================

import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# -------------------------------------------------------
# FIX WORKING DIRECTORY
# -------------------------------------------------------
# This makes sure Python uses the folder where this script exists
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("STEP 2: Training CNN Model")
print("=" * 60)

# -------------------------------------------------------
# CHECK DATASET EXISTS
# -------------------------------------------------------
required_folders = [
    "dataset/training_set/cats",
    "dataset/training_set/dogs",
    "dataset/test_set/cats",
    "dataset/test_set/dogs"
]

for folder in required_folders:

    # Check folder exists
    if not os.path.exists(folder):
        print(f"\n❌ ERROR: Folder not found -> {folder}")
        print("\nPlease check your dataset folder structure.")
        print("\nRequired structure:")
        print("""
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
│
└── test_set/
    ├── cats/
    └── dogs/
""")
        sys.exit()

    # Count images
    count = len(os.listdir(folder))

    # Check folder is not empty
    if count == 0:
        print(f"\n ERROR: No images found in -> {folder}")
        sys.exit()

    print(f" {folder}: {count} images found")

# -------------------------------------------------------
# IMAGE PREPROCESSING & AUGMENTATION
# -------------------------------------------------------
print("\n Setting up Image Preprocessing...")

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# Test data preprocessing
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# -------------------------------------------------------
# LOAD TRAINING DATA
# -------------------------------------------------------
print("\n Loading Training Dataset...")

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# -------------------------------------------------------
# LOAD TEST DATA
# -------------------------------------------------------
print("\n Loading Test Dataset...")

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# -------------------------------------------------------
# SAVE CLASS LABELS
# -------------------------------------------------------
class_indices = training_set.class_indices

print("\n Class Labels:")
print(class_indices)

with open("class_labels.json", "w") as f:
    json.dump(class_indices, f)

print(" Class labels saved to class_labels.json")

# -------------------------------------------------------
# BUILD CNN MODEL
# -------------------------------------------------------
print("\n Building CNN Model...")

cnn = Sequential()

# =====================================================
# FIRST CONVOLUTION BLOCK
# =====================================================
cnn.add(
    Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        padding='same',
        input_shape=[64, 64, 3]
    )
)

cnn.add(BatchNormalization())

cnn.add(
    MaxPooling2D(
        pool_size=2,
        strides=2
    )
)

# =====================================================
# SECOND CONVOLUTION BLOCK
# =====================================================
cnn.add(
    Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same'
    )
)

cnn.add(BatchNormalization())

cnn.add(
    MaxPooling2D(
        pool_size=2,
        strides=2
    )
)

# =====================================================
# THIRD CONVOLUTION BLOCK
# =====================================================
cnn.add(
    Conv2D(
        filters=128,
        kernel_size=3,
        activation='relu',
        padding='same'
    )
)

cnn.add(BatchNormalization())

cnn.add(
    MaxPooling2D(
        pool_size=2,
        strides=2
    )
)

# =====================================================
# FLATTEN LAYER
# =====================================================
cnn.add(Flatten())

# =====================================================
# FULLY CONNECTED LAYERS
# =====================================================
cnn.add(Dense(units=256, activation='relu'))
cnn.add(Dropout(0.5))

cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dropout(0.3))

# =====================================================
# OUTPUT LAYER
# =====================================================
cnn.add(Dense(units=1, activation='sigmoid'))

# -------------------------------------------------------
# COMPILE MODEL
# -------------------------------------------------------
print("\n Compiling Model...")

cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(" Model compiled successfully!")

# -------------------------------------------------------
# SHOW MODEL SUMMARY
# -------------------------------------------------------
print("\n Model Summary:")
cnn.summary()

# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------
print("\n Setting up Callbacks...")

checkpoint = ModelCheckpoint(
    'animal_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------
print("\n Training Started...")
print("This may take several minutes depending on your PC.")
print("=" * 60)

history = cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=30,
    callbacks=[checkpoint, early_stop]
)

# -------------------------------------------------------
# SAVE FINAL MODEL
# -------------------------------------------------------
print("\n Saving Final Model...")

cnn.save("animal_model.h5")

print(" Final model saved as: animal_model.h5")
print(" Best model saved as: animal_model_best.h5")

# -------------------------------------------------------
# EVALUATE MODEL
# -------------------------------------------------------
print("\n Evaluating Model...")

test_loss, test_accuracy = cnn.evaluate(test_set)

print(f"\n Test Accuracy: {test_accuracy * 100:.2f}%")
print(f" Test Loss: {test_loss:.4f}")

# -------------------------------------------------------
# PLOT TRAINING RESULTS
# -------------------------------------------------------
print("\n Generating Graphs...")

plt.figure(figsize=(12, 5))

# Accuracy Graph
plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Graph
plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save graph image
plt.savefig('training_results.png')

# Show graphs
plt.show()

print(" Training graphs saved as: training_results.png")

# -------------------------------------------------------
# TRAINING COMPLETE
# -------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print("\nNext Step:")
print("Run Step3_predict.py to classify new images.")