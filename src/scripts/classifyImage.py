from src import classes
import numpy as np
import json
import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = 5_000_000_000

def classifyImage(imageStream):

    image = np.fromfile(imageStream, np.uint8)
    imageStream = io.BytesIO(image)
    image = Image.open(imageStream)
    image = np.array(image)


    ovarianCancerModel = classes.CNNBoostModel(
        classes= 5,
        input_width= 250,
        input_height= 250,
    )
    ovarianCancerModel.load(path="./src/models/")
    prediction = ovarianCancerModel.predict(image=image)

    response = {
        "CC": ("Positive" if prediction[0][0] == 1 else "Negative"),
        "EC": ("Positive" if prediction[0][1] == 1 else "Negative"),
        "LGSC": ("Positive" if prediction[0][2] == 1 else "Negative"),
        "HGSC": ("Positive" if prediction[0][3] == 1 else "Negative"),
        "MC": ("Positive" if prediction[0][4] == 1 else "Negative")
    }
    return(json.dumps(response))