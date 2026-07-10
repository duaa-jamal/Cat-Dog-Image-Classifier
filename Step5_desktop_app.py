# ============================================================
# STEP 5: DESKTOP CLASSIFIER APP (Assignment Part a)
# A GUI application to classify cat/dog images
# Run: python Step5_desktop_app.py
# ============================================================

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import json
import os

# -------------------------------------------------------
# CHECK DEPENDENCIES
# -------------------------------------------------------
try:
    from PIL import Image, ImageTk
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image, ImageTk

# -------------------------------------------------------
# LOAD MODEL & LABELS
# -------------------------------------------------------
MODEL_PATH = "animal_model_best.h5"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "animal_model.h5"

model = None
class_indices = {}
index_to_class = {}

def load_resources():
    global model, class_indices, index_to_class
    try:
        model = load_model(MODEL_PATH)
        with open("class_labels.json", "r") as f:
            class_indices = json.load(f)
        index_to_class = {v: k for k, v in class_indices.items()}
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Could not load model!\nRun Step2_train_model.py first.\n\nError: {e}")
        return False

# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------
def predict_image(img_path):
    """Takes image path, returns (animal_name, cat_%, dog_%)"""
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array, verbose=0)
    prediction_value = float(result[0][0])

    if prediction_value > 0.5:
        predicted_index = 1
    else:
        predicted_index = 0

    animal = index_to_class[predicted_index].capitalize()
    dog_pct = prediction_value * 100
    cat_pct = (1 - prediction_value) * 100

    return animal, cat_pct, dog_pct

# -------------------------------------------------------
# BUILD GUI
# -------------------------------------------------------
class AnimalClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐾 Animal Classifier - Cat vs Dog Detector")
        self.root.geometry("600x700")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        self.build_ui()

    def build_ui(self):
        # ----- Title -----
        title_frame = tk.Frame(self.root, bg="#16213e", pady=15)
        title_frame.pack(fill="x")

        tk.Label(
            title_frame,
            text="🐾 Animal Classifier",
            font=("Helvetica", 22, "bold"),
            fg="#e94560",
            bg="#16213e"
        ).pack()

        tk.Label(
            title_frame,
            text="CNN-Based Cat vs Dog Detection System",
            font=("Helvetica", 11),
            fg="#a8a8b3",
            bg="#16213e"
        ).pack()

        # ----- Image Display -----
        img_frame = tk.Frame(self.root, bg="#0f3460", padx=10, pady=10)
        img_frame.pack(padx=20, pady=15, fill="x")

        self.image_label = tk.Label(
            img_frame,
            text=" No Image Selected\n\nClick 'Browse Image' to select\na cat or dog image",
            font=("Helvetica", 12),
            fg="#a8a8b3",
            bg="#0f3460",
            width=40,
            height=12,
            relief="flat"
        )
        self.image_label.pack()

        # ----- Buttons -----
        btn_frame = tk.Frame(self.root, bg="#1a1a2e")
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="  Browse Image",
            command=self.browse_image,
            font=("Helvetica", 12, "bold"),
            bg="#e94560",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="  Classify",
            command=self.classify,
            font=("Helvetica", 12, "bold"),
            bg="#0f3460",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="  Reset",
            command=self.reset,
            font=("Helvetica", 12),
            bg="#333",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        ).pack(side="left", padx=10)

        # ----- Result Box -----
        result_frame = tk.Frame(self.root, bg="#16213e", padx=20, pady=15)
        result_frame.pack(padx=20, pady=10, fill="x")

        tk.Label(
            result_frame,
            text="RESULT",
            font=("Helvetica", 10),
            fg="#a8a8b3",
            bg="#16213e"
        ).pack(anchor="w")

        self.result_label = tk.Label(
            result_frame,
            text="—",
            font=("Helvetica", 30, "bold"),
            fg="#e94560",
            bg="#16213e"
        )
        self.result_label.pack(pady=5)

        # ----- Confidence Bars -----
        bar_frame = tk.Frame(self.root, bg="#1a1a2e", padx=20)
        bar_frame.pack(fill="x", padx=20)

        # Cat bar
        tk.Label(bar_frame, text="🐱 Cat", font=("Helvetica", 11), fg="white", bg="#1a1a2e").pack(anchor="w")
        self.cat_bar = ttk.Progressbar(bar_frame, length=500, mode='determinate')
        self.cat_bar.pack(fill="x", pady=2)
        self.cat_pct_label = tk.Label(bar_frame, text="0%", fg="#a8a8b3", bg="#1a1a2e", font=("Helvetica", 10))
        self.cat_pct_label.pack(anchor="e")

        # Dog bar
        tk.Label(bar_frame, text="🐶 Dog", font=("Helvetica", 11), fg="white", bg="#1a1a2e").pack(anchor="w")
        self.dog_bar = ttk.Progressbar(bar_frame, length=500, mode='determinate')
        self.dog_bar.pack(fill="x", pady=2)
        self.dog_pct_label = tk.Label(bar_frame, text="0%", fg="#a8a8b3", bg="#1a1a2e", font=("Helvetica", 10))
        self.dog_pct_label.pack(anchor="e")

        # ----- Status bar -----
        self.status = tk.Label(
            self.root,
            text="Ready. Load a model and browse an image.",
            font=("Helvetica", 9),
            fg="#a8a8b3",
            bg="#0f3460",
            anchor="w",
            padx=10,
            pady=5
        )
        self.status.pack(fill="x", side="bottom")

        # Store current image path
        self.current_image_path = None

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if path:
            self.current_image_path = path
            self.display_image(path)
            self.status.config(text=f"Image loaded: {os.path.basename(path)}")
            self.result_label.config(text="—")
            self.cat_bar['value'] = 0
            self.dog_bar['value'] = 0

    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo    # Keep reference

    def classify(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please browse and select an image first!")
            return

        if model is None:
            messagebox.showerror("No Model", "Model not loaded. Run Step2_train_model.py first!")
            return

        self.status.config(text="🔍 Classifying...")
        self.root.update()

        try:
            animal, cat_pct, dog_pct = predict_image(self.current_image_path)

            self.result_label.config(
                text=f"{'🐱' if animal == 'Cats' else '🐶'} {animal.upper()} DETECTED",
                fg="#4ecca3" if animal == "Cats" else "#e94560"
            )

            self.cat_bar['value'] = cat_pct
            self.dog_bar['value'] = dog_pct
            self.cat_pct_label.config(text=f"{cat_pct:.1f}%")
            self.dog_pct_label.config(text=f"{dog_pct:.1f}%")

            confidence = max(cat_pct, dog_pct)
            self.status.config(text=f" Done! Confidence: {confidence:.1f}%")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status.config(text=" Error during classification")

    def reset(self):
        self.current_image_path = None
        self.image_label.config(
            image="",
            text=" No Image Selected\n\nClick 'Browse Image' to select\na cat or dog image"
        )
        self.result_label.config(text="—", fg="#e94560")
        self.cat_bar['value'] = 0
        self.dog_bar['value'] = 0
        self.cat_pct_label.config(text="0%")
        self.dog_pct_label.config(text="0%")
        self.status.config(text="Reset. Browse a new image.")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalClassifierApp(root)

    # Load model on startup
    if load_resources():
        app.status.config(text=" Model loaded. Browse an image to classify!")
    root.mainloop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
