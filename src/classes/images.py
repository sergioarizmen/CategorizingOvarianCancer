import cv2
import numpy as np
import os

class Image():
    def __init__(self, image=None) -> None:
        self._image = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
        self._imagePath = os.path.join('./images/', image.filename)
    
    def imageScale(self) -> None:
        self._image = cv2.resize(self._image, (800, 800))

    def safeToImages(self) -> None:
        cv2.imwrite(self._imagePath, self._image)
    
    def getDimensions(self) -> dict:
        height, width, channels = self._image.shape
        return { "width": width, "height": height}