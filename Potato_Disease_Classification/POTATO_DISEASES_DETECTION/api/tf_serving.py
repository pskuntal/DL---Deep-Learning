import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf


app = FastAPI()
endpoint = "http://localhost:8502/v1/models/Potato_Dieseases_Detection:predict"

CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


@app.get("/ping")
async def ping():
    return "Yello, I think yI m y'live"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    requests.post(endpoint, json=json_data)
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)