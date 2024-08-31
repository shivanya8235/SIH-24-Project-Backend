from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from keras.models import model_from_json
import uvicorn

app = FastAPI()

# Load the pre-trained model from joblib
model_path = 'C:/Users/rajya/Documents/Agri/model_data.joblib'
model_dict = joblib.load(model_path)

# Extract architecture and weights
architecture = model_dict['architecture']
weights = model_dict['weights']

# Reconstruct the model using the architecture
model = model_from_json(architecture)
model.set_weights(weights)

# Define class names as per your model
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

class ImageRequest(BaseModel):
    url: str

@app.post("/predict/")
async def predict(request: ImageRequest):
    try:
        # Fetch the image from the provided URL
        response = requests.get(request.url)
        response.raise_for_status()
        
        # Load the image
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')  # Ensure image is in RGB format
        
        # Preprocess the image
        image = image.resize((256, 256))  # Resize image to match model input dimensions
        image_array = np.array(image) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(image_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
