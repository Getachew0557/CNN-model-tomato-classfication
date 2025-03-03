from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join("saved-models", "model.keras")
MODEL = keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ['potato___Early_blight', 'potato___Late_blight', 'potato___healthy', 'unknown']  # Added 'unknown'

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        try:
            image = Image.open(file).convert('RGB')
            processed_image = preprocess_image(image)

            # Predict using the model
            predictions = MODEL.predict(processed_image)
            confidence = float(np.max(predictions[0]))
            second_confidence = float(np.partition(predictions[0], -2)[-2])  # Get the second highest confidence
            confidence_threshold = 0.6
            confidence_difference = confidence - second_confidence  # Calculate the difference



            # Set a confidence threshold for OOD detection
            if confidence < confidence_threshold or confidence_difference < 0.1:  # Check if the difference is small
                return render_template("index.html", prediction="unknown", confidence=confidence)  # Classify as unknown

            
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            return render_template("index.html", prediction=predicted_class, confidence=confidence)
        
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
