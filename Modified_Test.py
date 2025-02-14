import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import urllib.request

import os
# Load the saved models
cnn_model = load_model('cnn_model.h5')
knn_model = joblib.load('knn_model_cnn_features.pkl')

# Image preprocessing function
def preprocess_image(img_path, target_size=(50, 50)):
    try:
        img_array = cv2.imread(img_path, 1)
        img_array = cv2.medianBlur(img_array, 1)
        resized_array = cv2.resize(img_array, target_size)
        normalized_array = resized_array / 255.0
        return normalized_array
    except Exception as e:
        print(e)
        messagebox.showerror("Error", f"Failed to preprocess image: {e}")
        return None

# Predict function
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return

    # Add batch dimension for CNN input
    processed_img = np.expand_dims(processed_img, axis=0)

    # Extract features using CNN
    cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-5].output)
    features = cnn_feature_extractor.predict(processed_img)

    # Flatten features for KNN input
    features_flattened = features.reshape(1, -1)

    # Predict using KNN
    prediction = knn_model.predict(features_flattened)

    # Map prediction to class name
    class_index = int(prediction[0])
    class_name = subfolders[class_index] if class_index < len(subfolders) else "Unknown"

    return class_name

# Tkinter GUI
def open_file():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

        # Predict and display result
        result = predict_image(file_path)
        if result:
            result_label.config(text=f"Prediction: {result}")
            
            if result in "Alterneria_Mango":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, improve air circulation, avoid overhead watering.\nChemical Control: Use copper-based fungicides (Copper oxychloride, Bordeaux mixture) preventively; apply Mancozeb, Propiconazole, or Carbendazim during flowering; use hot water treatment (50°C for 15 minutes) post-harvest.\nBiological Control: Use Trichoderma biofungicides and neem oil spray.\nNutrient Management: Apply potassium, calcium, and organic compost for plant health.")
            if result in "Anthracnose_Mango":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, improve air circulation, avoid overhead watering, and use resistant varieties.\nChemical Control: Apply copper-based fungicides (Copper oxychloride, Bordeaux mixture) preventively; use systemic fungicides (Carbendazim, Azoxystrobin) during flowering; dip harvested mangoes in hot water (50°C for 15 minutes).\nBiological Control: Use Trichoderma biofungicides and neem oil spray for natural protection.\nNutrient Management: Apply potassium and calcium for immunity and improve soil health with compost.")
            if result in "Black Mould Rot_Mango":
                messagebox.showinfo("Remedy", "Cultural Control: Harvest at the right stage, ensure proper ventilation, and avoid fruit damage.\nChemical Control: Apply Carbendazim or Thiophanate-methyl pre-harvest; use hot water treatment (52°C for 5 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium for stronger fruit skin and maintain nitrogen balance.")
            if result in "Stem and Rot_Mango":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected branches, improve air circulation, harvest carefully, and store properly.\nChemical Control: Apply Carbendazim or Thiophanate-methyl pre-harvest; use hot water treatment (52°C for 5 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium for stronger fruit tissues and avoid excessive nitrogen.")
            if result in "Anthracnose_Guava":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, improve air circulation, avoid overhead irrigation, and use disease-free planting material.\nChemical Control: Apply Copper oxychloride or Bordeaux mixture preventively; use Carbendazim or Azoxystrobin during flowering; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply potassium and calcium for resistance and use compost for soil health.")
            if result in "Phytopthora_Guava":
                messagebox.showinfo("Remedy", "Cultural Control: Improve drainage, prune infected parts, remove fallen debris, and avoid overwatering.\nChemical Control: Use Metalaxyl or Fosetyl-Al as a soil drench; apply Copper oxychloride or Mancozeb as foliar sprays; use hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Apply Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for immunity and use compost for soil health.")
            if result in "Scab_Guava":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, improve air circulation, avoid overhead irrigation, and remove fallen debris.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Azoxystrobin during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for resistance and use compost for soil health.")
            if result in "Styler and Root_Guava":
                messagebox.showinfo("Remedy", "Cultural Control: Improve soil drainage, prune infected parts, maintain proper spacing, and avoid overwatering.\nChemical Control: Apply Metalaxyl or Fosetyl-Al as a soil drench; use Copper oxychloride or Mancozeb sprays; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for root health and use compost for soil improvement.")
            if result in "Blackspot_Orange":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, improve air circulation, avoid overhead irrigation, and remove fallen debris.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Azoxystrobin during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for resistance and use compost for soil health.")
            if result in "Canker_Orange":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, avoid mechanical injury, improve spacing and airflow, and remove fallen debris.\nChemical Control: Apply Copper hydroxide or Copper oxychloride preventively; use Streptomycin or Oxytetracycline during flowering and fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Bacillus subtilis, Pseudomonas fluorescens, or neem oil spray.\nNutrient Management: Apply potassium, calcium, magnesium, and use organic fertilizers for tree health.")
            if result in "Greening_Orange":
                messagebox.showinfo("Remedy", "Cultural Control: Remove infected trees, control psyllid vectors (Diaphorina citri), maintain orchard hygiene, and avoid spreading infected plant material.\nChemical Control: Use insecticides like Imidacloprid or Thiamethoxam for psyllid control; apply Streptomycin or Oxytetracycline to manage infection; use copper-based fungicides for secondary infections.\nBiological Control: Release natural predators like Tamarixia radiata and use Bacillus thuringiensis for psyllid control.\nNutrient Management: Apply balanced fertilizers with micronutrients (magnesium, zinc, calcium) to boost immunity and monitor soil health.")
            if result in "Blotch_Apple":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, maintain proper spacing, remove fallen debris, and avoid overhead irrigation.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Azoxystrobin during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply potassium and calcium for resistance and use compost for soil health.")
            if result in "Rot_Apple":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, remove decayed fruit and fallen debris, avoid mechanical injury, and ensure proper spacing and air circulation.\nChemical Control: Apply Mancozeb or Captan pre-harvest; use hot water treatment (52°C for 10 minutes) post-harvest; apply fungicidal wax coatings.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply potassium, calcium, magnesium, and use compost for soil health.")
            if result in "Scab_Apple":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, remove fallen leaves and fruit, maintain proper spacing, and avoid overhead irrigation.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Myclobutanil during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium, potassium, magnesium, and use organic fertilizers for soil health.")
            if result in "Canker_Lemon":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, avoid mechanical injuries, maintain proper spacing, and remove fallen leaves and fruit.\nChemical Control: Apply Copper hydroxide or Copper oxychloride preventively; use Streptomycin or Oxytetracycline during flowering and fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Bacillus subtilis, Pseudomonas fluorescens, or neem oil spray.\nNutrient Management: Apply potassium, calcium, magnesium, and use organic fertilizers for soil health.")
            if result in "Mold_Lemon":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, maintain proper spacing, remove fallen fruits and debris, and avoid over-irrigation.\nChemical Control: Apply Mancozeb or Copper oxychloride pre-harvest; use hot water treatment (52°C for 10 minutes) post-harvest; apply fungicidal wax coatings.Biological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium, potassium, and use compost for soil health.")
            if result in "Scab_Lemon":
                messagebox.showinfo("Remedy", "Cultural Control: Prune infected parts, remove fallen leaves and fruit, maintain proper spacing, and avoid overhead irrigation.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Myclobutanil during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium, potassium, magnesium, and use organic fertilizers for soil health.")    
    except Exception as e:
        print(e)
        messagebox.showerror("Error", f"Failed to open image: {e}")

# Load subfolder names
try:
    data_dir = "data"  # Ensure the same directory used in training
    subfolders = os.listdir(data_dir)
except Exception as e:
    subfolders = []
    print(f"Error loading subfolders: {e}")

# Initialize Tkinter window
root = tk.Tk()
root.title("Image Classification using KNN")

# Create and place widgets
open_button = tk.Button(root, text="Open Image", command=open_file)
open_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="Prediction: None", font=("Arial", 14))
result_label.pack(pady=10)

# Run Tkinter main loop
root.mainloop()
