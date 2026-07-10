# ============================================================
# STEP 4: EVALUATE MODEL (For Assignment Metrics)
# Calculates Accuracy, Precision, Recall, F1-Score
# ============================================================

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

print("=" * 60)
print("STEP 4: Model Evaluation & Metrics")
print("=" * 60)

# -------------------------------------------------------
# LOAD MODEL & LABELS
# -------------------------------------------------------
model = load_model("animal_model_best.h5")

with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

# {'cats': 0, 'dogs': 1}
index_to_class = {v: k for k, v in class_indices.items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]

print(f"Classes: {class_names}")

# -------------------------------------------------------
# LOAD TEST DATA
# -------------------------------------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False    # IMPORTANT: Don't shuffle for evaluation
)

# -------------------------------------------------------
# GET PREDICTIONS
# -------------------------------------------------------
print("\n Running predictions on entire test set...")
predictions_raw = model.predict(test_set, verbose=1)

# Convert probabilities to class labels (0 or 1)
predictions = (predictions_raw > 0.5).astype(int).flatten()

# Get true labels
true_labels = test_set.classes

# -------------------------------------------------------
# CALCULATE METRICS
# -------------------------------------------------------
accuracy  = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall    = recall_score(true_labels, predictions)
f1        = f1_score(true_labels, predictions)

print("\n" + "=" * 60)
print(" EVALUATION METRICS (For Assignment)")
print("=" * 60)
print(f" Accuracy  : {accuracy  * 100:.2f}%")
print(f" Precision : {precision * 100:.2f}%")
print(f" Recall    : {recall    * 100:.2f}%")
print(f" F1-Score  : {f1        * 100:.2f}%")
print("=" * 60)

print("\n Detailed Classification Report:")
print(classification_report(true_labels, predictions, target_names=class_names))

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix\n(How many cats/dogs were correctly identified)', fontsize=13)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print(" Confusion matrix saved as confusion_matrix.png")

# -------------------------------------------------------
# EXPLAIN METRICS FOR ASSIGNMENT WRITING
# -------------------------------------------------------
print("\n" + "=" * 60)
print(" METRICS EXPLANATION (Use in your assignment)")
print("=" * 60)
print(f"""
ACCURACY ({accuracy*100:.2f}%):
  Out of ALL images tested, the model correctly classified
  {accuracy*100:.2f}% of them.

PRECISION ({precision*100:.2f}%):
  When the model says "this is a dog", it is correct
  {precision*100:.2f}% of the time.
  (Measures: quality of positive predictions)

RECALL ({recall*100:.2f}%):
  Out of ALL actual dogs in the test set, the model
  correctly found {recall*100:.2f}% of them.
  (Measures: how many real cases were caught)

F1-SCORE ({f1*100:.2f}%):
  A balanced average of Precision and Recall.
  Higher is better. Ideal value = 100%.

CONFUSION MATRIX:
  Shows exact counts of:
  - True Positives  (Dog correctly identified as Dog)
  - True Negatives  (Cat correctly identified as Cat)
  - False Positives (Cat wrongly identified as Dog)
  - False Negatives (Dog wrongly identified as Cat)
""")
