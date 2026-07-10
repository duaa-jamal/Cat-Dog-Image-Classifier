# 🐾 Animal Classifier (Cat vs Dog CNN)
## Complete Beginner Guide

---

## 📁 Files in This Project

| File | Purpose |
|------|---------|
| `Step1_download_dataset.py` | Creates folder structure + dataset instructions |
| `Step2_train_model.py` | Trains the CNN model |
| `Step3_predict.py` | Classifies a single image |
| `Step4_evaluate_model.py` | Calculates Accuracy, Precision, Recall |
| `Step5_desktop_app.py` | Desktop GUI application |

---

## 🚀 HOW TO RUN (Follow these steps IN ORDER)

---

### ✅ STEP 1: Install Required Libraries

Open **Spyder Terminal** (or Anaconda Prompt) and run:

```
pip install tensorflow keras numpy matplotlib pillow scikit-learn seaborn
```

---

### ✅ STEP 2: Create Folder Structure

Run this in Spyder:
```
python Step1_download_dataset.py
```

This creates:
```
dataset/
  training_set/
    cats/    ← put training cat images here
    dogs/    ← put training dog images here
  test_set/
    cats/    ← put test cat images here
    dogs/    ← put test dog images here
```

---

### ✅ STEP 3: Download Dataset

**Go to this link:**
👉 https://www.kaggle.com/datasets/tongpython/cat-and-dog

1. Create a free Kaggle account
2. Click **Download**
3. Extract the ZIP file
4. Copy images into the folders created above

**Minimum images needed:**
- training_set/cats/ → at least 200 cat images
- training_set/dogs/ → at least 200 dog images
- test_set/cats/ → at least 50 cat images
- test_set/dogs/ → at least 50 dog images

> 💡 More images = Better accuracy!

---

### ✅ STEP 4: Train the Model

In Spyder, open `Step2_train_model.py` and press **Run (F5)**

⏱️ This takes 10-30 minutes depending on your PC.

When done you'll see:
- `animal_model.h5` - your trained model
- `animal_model_best.h5` - best version of your model
- `class_labels.json` - your class labels
- `training_results.png` - accuracy/loss graphs

---

### ✅ STEP 5: Test a Single Image

1. Open `Step3_predict.py`
2. Change this line to your image path:
   ```python
   IMAGE_PATH = "sample.jpg"   # Change this!
   ```
3. Press **Run (F5)**

You'll see output like:
```
🎯 PREDICTION RESULT
🐾 Detected Animal: CATS
🐱 Cat Probability: 87.3%
🐶 Dog Probability: 12.7%
📈 Confidence Level: High Confidence ✅
```

---

### ✅ STEP 6: Get Assignment Metrics

Run `Step4_evaluate_model.py` to get:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

### ✅ STEP 7: Run Desktop App

Run `Step5_desktop_app.py` to open the GUI app!

---

## 🔧 FIXING THE ORIGINAL PROBLEM (Why cats were predicted as dogs)

**The bug was in the original `predict.py`:**

```python
# ❌ WRONG (original code)
if result[0][0] > 0.5:
    print("Cat Detected")   # This is actually a DOG!
else:
    print("Dog Detected")   # This is actually a CAT!
```

**Why was it wrong?**
- Keras reads folders alphabetically: `cats=0`, `dogs=1`
- Sigmoid output close to **1** means **dogs**
- Sigmoid output close to **0** means **cats**
- So `result > 0.5` means DOG, not cat!

```python
# ✅ CORRECT (fixed code)
if result[0][0] > 0.5:
    print("Dog Detected")   # ✅ Correct!
else:
    print("Cat Detected")   # ✅ Correct!
```

---

## 📊 Assignment Answers

### Proposed Methodology
**Convolutional Neural Network (CNN)** with:
- 3 Convolutional layers (32, 64, 128 filters)
- BatchNormalization for training stability
- MaxPooling for dimensionality reduction
- Dropout (0.5) to prevent overfitting
- Sigmoid output for binary classification

### Dataset
- **Source:** Kaggle - "Cat and Dog" by tongpython
- **Link:** https://www.kaggle.com/datasets/tongpython/cat-and-dog
- **Size:** ~25,000 images of cats and dogs

### Tools/Environment
- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- Scikit-learn (metrics)
- Spyder IDE (Anaconda)

### Society Benefits
- Can help animal shelters automate animal identification
- Can assist visually impaired users in identifying animals
- Foundation for more complex wildlife monitoring systems
- Reduces manual effort in sorting animal image databases

---

## ❓ TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install tensorflow` |
| Low accuracy (<70%) | Add more images and retrain |
| Model always predicts same class | Check class_labels.json mapping |
| `class_labels.json not found` | Run Step2 first |
| Desktop app doesn't open | Install Pillow: `pip install Pillow` |
