from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL=tf.keras.models.load_model("../models/1") # here .. is used to return to the parent directory
classes_names=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy']

app=FastAPI()
# this is just for testing purpose
@app.get("/ping")
async def ping():
    return  "Succesfully installed fast api"
'''Actual server building starts here '''

def read_file_as_image(data) -> np.ndarray: # this function returns array of incoming file which is in bytes
    image = np.array(Image.open(BytesIO(data)))
    return image

# Now we build the prediction program
@app.post("/prediction")
async def prediction(file: UploadFile = File(...)):# this takes input of type File(uploadfile) and default value is File(...)
    image=read_file_as_image(await file.read()) # it will read bytes of incoming file from user

    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = classes_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8080)
