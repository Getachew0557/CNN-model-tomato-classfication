from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join("saved-models", "tomato.keras")
MODEL = keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

CONFIDENCE_THRESHOLD = 0.6  # Define a threshold for classification

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize
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
            softmax_probs = keras.activations.softmax(predictions[0]).numpy()  # Apply softmax
            
            # Get the highest probability and corresponding class
            max_prob = float(np.max(softmax_probs))
            predicted_class = CLASS_NAMES[np.argmax(softmax_probs)]

            # If max probability is below the threshold, classify as "unknown"
            if max_prob < CONFIDENCE_THRESHOLD:
                return render_template("index.html", prediction="unknown", confidence=max_prob)

            return render_template("index.html", prediction=predicted_class, confidence=max_prob)
        
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
