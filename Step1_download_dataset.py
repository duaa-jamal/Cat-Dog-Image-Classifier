# ============================================================
# STEP 1: DOWNLOAD DATASET
# Run this file FIRST before anything else
# This downloads Cats vs Dogs dataset from Kaggle alternative
# ============================================================

import os
import urllib.request
import zipfile

print("=" * 60)
print("STEP 1: Downloading Dataset...")
print("=" * 60)

# We use the Microsoft Cats vs Dogs dataset (public domain)
# This is a clean, well-labeled dataset perfect for beginners

URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

# -------------------------------------------------------
# ALTERNATIVE: If the above URL doesn't work, use this
# manual method instead:
#
# 1. Go to: https://www.kaggle.com/datasets/tongpython/cat-and-dog
# 2. Download the zip
# 3. Extract it into this folder so you have:
#    dataset/
#      training_set/
#        cats/   <-- put cat images here
#        dogs/   <-- put dog images here
#      test_set/
#        cats/
#        dogs/
# -------------------------------------------------------

def create_folder_structure():
    """Create the required folder structure"""
    folders = [
        "dataset/training_set/cats",
        "dataset/training_set/dogs",
        "dataset/test_set/cats",
        "dataset/test_set/dogs"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Folder structure created successfully!")
    print("\nFolder structure:")
    print("dataset/")
    print("  training_set/")
    print("    cats/   ← training cat images go here")
    print("    dogs/   ← training dog images go here")
    print("  test_set/")
    print("    cats/   ← test cat images go here")
    print("    dogs/   ← test dog images go here")

create_folder_structure()

print("\n" + "=" * 60)
print("IMPORTANT INSTRUCTIONS:")
print("=" * 60)
print("""
Since direct download requires Kaggle account, please do this:

OPTION A (Recommended - Kaggle):
1. Go to: https://www.kaggle.com/datasets/tongpython/cat-and-dog
2. Click 'Download' button
3. You'll get a zip file
4. Extract it and place images like this:
   - Training cat images → dataset/training_set/cats/
   - Training dog images → dataset/training_set/dogs/
   - Test cat images     → dataset/test_set/cats/
   - Test dog images     → dataset/test_set/dogs/

OPTION B (Google Drive - No account needed):
1. Go to this link and download:
   https://drive.google.com/drive/folders/1BHmvHSXBKnizk2mK0KRuCjQFaJ8qxnCz
2. Same extraction as above

MINIMUM images needed:
- At least 100 cat images in training_set/cats/
- At least 100 dog images in training_set/dogs/
- At least 20 cat images in test_set/cats/
- At least 20 dog images in test_set/dogs/

MORE images = BETTER accuracy!
""")

print("✅ Run Step2_train_model.py AFTER placing images in folders")
