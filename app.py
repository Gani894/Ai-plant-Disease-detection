from flask import Flask, render_template, request, session
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session

# Base directory for resolving paths reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your trained model
model = load_model(os.path.join(BASE_DIR, "plant_disease_model.h5"))

# Load class names from JSON file
with open(os.path.join(BASE_DIR, 'class_labels.json'), 'r') as f:
    class_names = json.load(f)

# Cure or treatment info (expanded for all classes)
cure_suggestions = {
    "Apple___Apple_scab": "Use fungicides like Captan or Mancozeb. Remove infected leaves and prune the tree regularly.",
    "Apple___Black_rot": "Remove infected fruits and leaves. Apply copper-based fungicide. Keep area dry.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees if possible. Apply fungicides like myclobutanil.",
    "Apple___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Blueberry___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicides. Improve air circulation around plants.",
    "Cherry_(including_sour)___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant varieties. Apply fungicides like azoxystrobin.",
    "Corn_(maize)___Common_rust_": "Use resistant hybrids and apply fungicides such as azoxystrobin.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops and apply fungicides. Remove infected debris.",
    "Corn_(maize)___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Grape___Black_rot": "Remove infected parts. Apply fungicides like mancozeb.",
    "Grape___Esca_(Black_Measles)": "Prune infected vines. No cure, focus on prevention.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper fungicides. Improve drainage.",
    "Grape___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees. Control psyllid vectors with insecticides.",
    "Peach___Bacterial_spot": "Apply copper fungicides. Use resistant varieties.",
    "Peach___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Pepper,_bell___Bacterial_spot": "Use copper fungicides. Avoid overhead watering.",
    "Pepper,_bell___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Potato___Early_blight": "Apply fungicides like chlorothalonil. Rotate crops.",
    "Potato___Late_blight": "Remove infected plants. Apply fungicides like mancozeb.",
    "Potato___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Raspberry___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Soybean___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Squash___Powdery_mildew": "Apply sulfur fungicides. Improve air circulation.",
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply fungicides.",
    "Strawberry___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free.",
    "Tomato___Bacterial_spot": "Use copper fungicides. Avoid wet leaves.",
    "Tomato___Early_blight": "Apply fungicides. Mulch to prevent soil splash.",
    "Tomato___Late_blight": "Remove infected plants immediately. Apply fungicides.",
    "Tomato___Leaf_Mold": "Improve ventilation. Apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove lower leaves. Apply fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap or neem oil.",
    "Tomato___Target_Spot": "Apply fungicides. Avoid overhead watering.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with insecticides. Use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants. Disinfect tools.",
    "Tomato___healthy": "No treatment needed 🌿 Your plant is healthy! Keep it well-watered and pest-free."
}


@app.route('/')
def home():
    history = session.get('image_history', [])
    return render_template('index.html', history=history)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No image uploaded")

    # Save uploaded file
    filename = file.filename or 'uploaded_image.jpg'
    static_folder = os.path.join(BASE_DIR, 'static')
    os.makedirs(static_folder, exist_ok=True)
    img_save_path = os.path.join(static_folder, filename)
    file.save(img_save_path)

    # Web-friendly path with forward slashes for HTML img src
    img_path = f"static/{filename}"

    # Preprocess image for model (MobileNet expects 224x224)
    img = image.load_img(img_save_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    predictions = model.predict(img_array)
    class_index = int(np.argmax(predictions))
    prediction = class_names[class_index]
    confidence = float(round(float(np.max(predictions)) * 100, 2))

    # Get cure suggestion
    cure = cure_suggestions.get(prediction, "Cure information not available.")

    # Add to history (keep last 3)
    history = session.get('image_history', [])
    history_entry = {
        'img_path': img_path,
        'prediction': prediction,
        'confidence': confidence,
        'cure': cure
    }
    history.append(history_entry)
    if len(history) > 3:
        history.pop(0)
    session['image_history'] = history

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           cure=cure,
                           img_path=img_path,
                           history=history)


if __name__ == '__main__':
    app.run(debug=True)
