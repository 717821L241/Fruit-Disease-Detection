from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
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
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Flask app setup
# Load models
cnn_model = load_model('cnn_model.h5')
knn_model = joblib.load('knn_model_cnn_features.pkl')

# GPT-2 setup

data_dir = "data"

class_names = os.listdir(data_dir)  # Load class names from the dataset

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Image preprocessing function


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def preprocess_image(img_path, target_size=(50, 50)):
    try:
        img_array = cv2.imread(img_path, 1)
        img_array = cv2.medianBlur(img_array, 1)
        resized_array = cv2.resize(img_array, target_size)
        normalized_array = resized_array / 255.0
        return normalized_array
    except Exception as e:
        print(e)
        return None


@app.route('/')
def index():
    return render_template('index.html')

# Predict function
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return "Error in preprocessing"

    processed_img = np.expand_dims(processed_img, axis=0)

    # Extract features using CNN
    cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-5].output)
    features = cnn_feature_extractor.predict(processed_img)

    features_flattened = features.reshape(1, -1)

    prediction = knn_model.predict(features_flattened)
    class_index = int(prediction[0])
    class_name = class_names[class_index]
    print(class_index)

    return class_name


@app.route('/predict', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform prediction
        result = predict_image(file_path)
        if result:
            global explanation
            if result in "Alterneria_Mango":
                explanation="Cultural Control: Prune infected parts, improve air circulation, avoid overhead watering.\nChemical Control: Use copper-based fungicides (Copper oxychloride, Bordeaux mixture) preventively; apply Mancozeb, Propiconazole, or Carbendazim during flowering; use hot water treatment (50°C for 15 minutes) post-harvest.\nBiological Control: Use Trichoderma biofungicides and neem oil spray.\nNutrient Management: Apply potassium, calcium, and organic compost for plant health."
            elif result in "Anthracnose_Mango":
                explanation="Cultural Control: Prune infected parts, improve air circulation, avoid overhead watering, and use resistant varieties.\nChemical Control: Apply copper-based fungicides (Copper oxychloride, Bordeaux mixture) preventively; use systemic fungicides (Carbendazim, Azoxystrobin) during flowering; dip harvested mangoes in hot water (50°C for 15 minutes).\nBiological Control: Use Trichoderma biofungicides and neem oil spray for natural protection.\nNutrient Management: Apply potassium and calcium for immunity and improve soil health with compost."
            elif result in "Black Mould Rot_Mango":
                explanation= "Cultural Control: Harvest at the right stage, ensure proper ventilation, and avoid fruit damage.\nChemical Control: Apply Carbendazim or Thiophanate-methyl pre-harvest; use hot water treatment (52°C for 5 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium for stronger fruit skin and maintain nitrogen balance."
            elif result in "Stem and Rot_Mango":
                explanation= "Cultural Control: Prune infected branches, improve air circulation, harvest carefully, and store properly.\nChemical Control: Apply Carbendazim or Thiophanate-methyl pre-harvest; use hot water treatment (52°C for 5 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium for stronger fruit tissues and avoid excessive nitrogen."
            elif result in "Anthracnose_Guava":
               explanation= "Cultural Control: Prune infected parts, improve air circulation, avoid overhead irrigation, and use disease-free planting material.\nChemical Control: Apply Copper oxychloride or Bordeaux mixture preventively; use Carbendazim or Azoxystrobin during flowering; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply potassium and calcium for resistance and use compost for soil health."
            elif result in "Phytopthora_Guava":
                explanation= "Cultural Control: Improve drainage, prune infected parts, remove fallen debris, and avoid overwatering.\nChemical Control: Use Metalaxyl or Fosetyl-Al as a soil drench; apply Copper oxychloride or Mancozeb as foliar sprays; use hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Apply Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for immunity and use compost for soil health."
            elif result in "Scab_Guava":
               explanation= "Cultural Control: Prune infected parts, improve air circulation, avoid overhead irrigation, and remove fallen debris.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Azoxystrobin during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for resistance and use compost for soil health."
            elif result in "Styler and Root_Guava":
                explanation= "Cultural Control: Improve soil drainage, prune infected parts, maintain proper spacing, and avoid overwatering.\nChemical Control: Apply Metalaxyl or Fosetyl-Al as a soil drench; use Copper oxychloride or Mancozeb sprays; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for root health and use compost for soil improvement."
            elif result in "Blackspot_Orange":
                explanation= "Cultural Control: Prune infected parts, improve air circulation, avoid overhead irrigation, and remove fallen debris.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Azoxystrobin during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium and potassium for resistance and use compost for soil health."
            elif result in "Canker_Orange":
                explanation= "Cultural Control: Prune infected parts, avoid mechanical injury, improve spacing and airflow, and remove fallen debris.\nChemical Control: Apply Copper hydroxide or Copper oxychloride preventively; use Streptomycin or Oxytetracycline during flowering and fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Bacillus subtilis, Pseudomonas fluorescens, or neem oil spray.\nNutrient Management: Apply potassium, calcium, magnesium, and use organic fertilizers for tree health."
            elif result in "Greening_Orange":
                explanation= "Cultural Control: Remove infected trees, control psyllid vectors (Diaphorina citri), maintain orchard hygiene, and avoid spreading infected plant material.\nChemical Control: Use insecticides like Imidacloprid or Thiamethoxam for psyllid control; apply Streptomycin or Oxytetracycline to manage infection; use copper-based fungicides for secondary infections.\nBiological Control: Release natural predators like Tamarixia radiata and use Bacillus thuringiensis for psyllid control.\nNutrient Management: Apply balanced fertilizers with micronutrients (magnesium, zinc, calcium) to boost immunity and monitor soil health."
            elif result in "Blotch_Apple":
                explanation= "Cultural Control: Prune infected parts, maintain proper spacing, remove fallen debris, and avoid overhead irrigation.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Azoxystrobin during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply potassium and calcium for resistance and use compost for soil health."
            elif result in "Rot_Apple":
                explanation= "Cultural Control: Prune infected parts, remove decayed fruit and fallen debris, avoid mechanical injury, and ensure proper spacing and air circulation.\nChemical Control: Apply Mancozeb or Captan pre-harvest; use hot water treatment (52°C for 10 minutes) post-harvest; apply fungicidal wax coatings.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply potassium, calcium, magnesium, and use compost for soil health."
            elif result in "Scab_Apple":
                explanation= "Cultural Control: Prune infected parts, remove fallen leaves and fruit, maintain proper spacing, and avoid overhead irrigation.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Myclobutanil during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium, potassium, magnesium, and use organic fertilizers for soil health."
            elif result in "Canker_Lemon":
                explanation= "Cultural Control: Prune infected parts, avoid mechanical injuries, maintain proper spacing, and remove fallen leaves and fruit.\nChemical Control: Apply Copper hydroxide or Copper oxychloride preventively; use Streptomycin or Oxytetracycline during flowering and fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Bacillus subtilis, Pseudomonas fluorescens, or neem oil spray.\nNutrient Management: Apply potassium, calcium, magnesium, and use organic fertilizers for soil health."
            elif result in "Mold_Lemon":
                explanation= "Cultural Control: Prune infected parts, maintain proper spacing, remove fallen fruits and debris, and avoid over-irrigation.\nChemical Control: Apply Mancozeb or Copper oxychloride pre-harvest; use hot water treatment (52°C for 10 minutes) post-harvest; apply fungicidal wax coatings.Biological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium, potassium, and use compost for soil health."
            elif result in "Scab_Lemon":
                explanation= "Cultural Control: Prune infected parts, remove fallen leaves and fruit, maintain proper spacing, and avoid overhead irrigation.\nChemical Control: Apply Copper oxychloride or Mancozeb preventively; use Tebuconazole or Myclobutanil during fruit development; hot water treatment (50°C for 10 minutes) post-harvest.\nBiological Control: Use Trichoderma harzianum, Bacillus subtilis, or neem oil spray.\nNutrient Management: Apply calcium, potassium, magnesium, and use organic fertilizers for soil health."
            else:
                explanation="Healthy"
                
            return render_template('index.html', label=result, control_measure=explanation, image_url=file_path)
        return redirect(url_for('index'))
        


if __name__ == '__main__':
    app.run(debug=True)
