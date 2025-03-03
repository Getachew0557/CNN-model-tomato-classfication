from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from scipy.spatial import distance
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("../saved-models/model.keras", compile=False)
MODEL = tf.keras.models.load_model("tomato/saved-models/model.keras", compile=False)

CLASS_NAMES = ['potato___Early_blight', 'potato___Late_blight', 'potato___healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')  
    image = image.resize((256, 256))  
    image = np.array(image) / 255.0  
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  

    # Calculate Mahalanobis distance for OOD detection
    distance = calculate_mahalanobis_distance(image, MODEL)
    threshold = 2.0  # Define a suitable threshold

    if distance > threshold:
        return {
            'class': 'Non-tomato leaf',
            'confidence': 0.0
        }
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return {
        'class': predicted_class,
        'confidence': confidence
    }

def calculate_mahalanobis_distance(image, model):
    # Placeholder for mean and covariance calculation
    mean = np.zeros((256, 256, 3))  # Adjust based on your training data
    cov = np.eye(3)  # Placeholder for covariance matrix
    image_flat = image.flatten()
    return distance.mahalanobis(image_flat, mean.flatten(), np.linalg.inv(cov))

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
