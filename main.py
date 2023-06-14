from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import io
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from places_data import places_data
import json

app = FastAPI()

# Load model only once when the application starts
model_path = "./model_transfer_learning.h5"
modelx = tf.saved_model.load(model_path)
dataset = pd.read_csv("_Resto_for_ImageRecog.csv")

def get_restaurant(prediction, dataset_resto):
    monas = "ChIJLbFk59L1aS4RyLzp4OHWKj0"
    museum_fatahillah = "ChIJfaWSQv8dai4RRQeMZy0D8BI"
    print("prediction: " + prediction)
    if prediction == "monas":
        place_id = monas
    elif prediction == "kotatua":
        place_id = museum_fatahillah

    nearest_resto = dataset_resto[dataset_resto["par_id"] == place_id].sort_values(["popularity", "distance_part_of_cluster"], ascending=[False, False])
    print("nearest_resto =")
    print(nearest_resto)

    top_20 = nearest_resto.sample(n=20)
    selection = top_20.sample(n=7).sort_values("popularity", ascending=False)

    return selection.to_json(orient="records")

@app.get("/")
async def home():
    return {"message": "Welcome to Image Recognition API!"}

@app.post("/predict")
async def upload_image(file: UploadFile = File(...)):
    try:
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

        detail = places_data.get(label)

        detail["restaurant"] = json.loads(get_restaurant(prediction = label, dataset_resto = dataset))

        if detail:
            return {
                "prediction": label,
                "detail": detail,
                "probability": probability
            }
        else:
            return {
                "prediction": label,
                "detail": None,
                "probability": probability
            }

    except Exception as e:
        return JSONResponse(
            status_code=401,
            content={"message": "Invalid image file.", "details": str(e)}
        )

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, log_level="info")