import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import tensorflow as tf
#from starlette.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["http://localhost:3000"]

# origins = [
#     "http://localhost",
#     "https://localhost:3000/",
# ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


prod_model = tf.keras.models.load_model("../Saved_Models/June_15_2022_17_08_07")

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
    predictions = prod_model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)