from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
import uvicorn
import tensorflow as tf
import io
from tensorflow.keras.preprocessing import image
import numpy as np

app = FastAPI()

# Load model only once when the application starts
model_path = "./model_transfer_learning.h5"
modelx = tf.saved_model.load(model_path)

@app.get("/")
async def home():
    return {"message": "Welcome to Rey's API!"}

@app.post("/predict")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    print("Loading Image")
    img = image.load_img(io.BytesIO(contents), target_size=(150, 150))
    # Importing the image to be predicted
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.

    print("Making a Prediction")
    # Prediction
    prediction = modelx(x)

    class_labels = ['ampera', 'gadang', 'gwk', 'kotatua', 'monas', 'ulundanu']
    label = class_labels[np.argmax(prediction)]
    probability = float(np.max(prediction))  # Convert numpy.float32 to Python float

    return {"prediction": label, "probability": probability}


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, log_level="info")
