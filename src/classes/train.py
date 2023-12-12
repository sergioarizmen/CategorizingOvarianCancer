import logging
import os

import PIL
import numpy as np
import pandas as pd

from cnn_boost_model import CNNBoostModel
from .exception_decorator import log_exception
from .logger import set_default_logging_configuration
from .resolve_path import resolve_path
from image_classification import get_classification

# Configure max image loading size.
PIL.Image.MAX_IMAGE_PIXELS = 5_000_000_000


# Configuration defaults.
PATH_CLASSIFICATION_FILE = '.hypothesis\\train.csv'
PATH_CLASSIFICATION_MAPPING_JSON = 'categories.json'
PATH_IMAGES_TRAIN = '.hypothesis\\images\\'
PATH_MODEL_SAVE = '.hypothesis\\model'

@log_exception
def train_with_images(model,
                      classification: pd.DataFrame,
                      images_directory_path: str
                      ):
    """
    Filter out the elements in the classification set that don't have a corresponding image file.
    :param model:
    :param classification: The classification to validate.
    :param images_directory_path: Path to the image files.
    :returns: A validated series of images.
    """

    # Iterate over all categorized images.
    for image_id, labels in classification.iterrows():
        # Get the image file path.
        image_path = resolve_path(os.path.join(images_directory_path, f'{image_id}.png'))

        # If the image file exists.
        if os.path.isfile(image_path):
            # Load the image.
            logging.info(f'Loading image from path: {image_path}')
            image = np.asarray(PIL.Image.open(image_path))

            # Train on the patches.
            model.train(image, labels.values)


# Run as main script.
if __name__ == '__main__':
    # Set default logging configuration.
    set_default_logging_configuration()

    # Set logging level to debug.
    logging.root.setLevel(10)

    # Create a model instance.
    model = CNNBoostModel(
        classes=5,
        input_height=250,
        input_width=250,
        epochs=1,
    )

    # Initialize the model from zero.
    # model.initialize()

    # Load the model from a directory.
    logging.info(f'Loading model from path: {PATH_MODEL_SAVE}')
    model.load(PATH_MODEL_SAVE)

    # Get the classification train set.
    classification = get_classification(
        PATH_CLASSIFICATION_FILE,
        PATH_CLASSIFICATION_MAPPING_JSON,
        PATH_IMAGES_TRAIN
    )

    # Train using the train set.
    train_with_images(model, classification, PATH_IMAGES_TRAIN)

    # Save the model.
    logging.info(f'Saving model to path: {PATH_MODEL_SAVE}')
    model.save(PATH_MODEL_SAVE)
