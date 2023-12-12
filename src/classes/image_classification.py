import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
from .exception_decorator import log_exception
from .resolve_path import resolve_path

@log_exception
def _load_classification_mapping(classification_mapping_path: str) -> dict[str, int]:
    """
    Load the classification mapping from a JSON file.
    :param classification_mapping_path: The path to the JSON classification mapping.
    """

    # With the JSON schema file open in read mode.
    with open(classification_mapping_path, 'r') as file:
        # Return the classification mapping file as a dictionary.
        return json.load(file)


@log_exception
def _load_image_classification(path: str) -> pd.Series:
    """
    Load the image classification.
    :param path: Path to the image classification.
    :returns: The corresponding classification.
    """

    # Read the classification as a dataframe.
    classification = pd.read_csv(path, index_col='image_id')

    # Keep only the classification column.
    classification = classification['label']

    # Return the loaded classification.
    return classification


@log_exception
def _filter_keep_valid_images(classification: pd.Series,
                              path: str
                              ) -> pd.Series:
    """
    Filter out the elements in the classification set that don't have a corresponding image file.
    :param classification: The classification to validate.
    :param path: Path to the image files.
    :returns: A validated series of images.
    """

    # Initialize an empty dict for the valid images.
    valid_images = {}

    # Iterate over all categorized images.
    for image_id, label in classification.items():
        # Get the image file path.
        image_path = os.path.join(path, f'{image_id}.png')

        # If the image file exists.
        if os.path.isfile(image_path):
            # Add the image to the array.
            valid_images[image_id] = label

    # Return the valid images.
    return pd.Series(valid_images, dtype=classification.dtype)


@log_exception
def _encode_classification(classification: pd.Series,
                           mapping: dict[str, int]
                           ) -> pd.Series:
    """
    Encode the classification names into integers using the given mapping.
    :param classification: The classification to encode.
    :param mapping: Map from the classification to the integer encode.
    :returns: The encoded classification.
    """

    # Encode the classification using the mapping.
    classification = classification.apply(lambda x: mapping[x])

    # Return the encoded classification.
    return classification


@log_exception
def _one_hot_encode_classification(classification: pd.Series,
                                   mapping: dict[str, int]
                                   ) -> pd.DataFrame:
    """
    One hot encode the classification.
    Get the encoded classification as a pandas dataframe with the classification map's names for columns.
    :param classification: The classification to one hot encode.
    :param mapping: Map from the classification to the integer encode.
    :return:
    """

    # One hot encode the categories into an encoded matrix (2d array).
    one_hot_encoded_array = tf.keras.utils.to_categorical(classification, num_classes=len(mapping))

    # Return the one hot encoded classification.
    return pd.DataFrame(one_hot_encoded_array, index=classification.index, columns=list(mapping.keys()))


@log_exception
def get_classification(classification_path: str,
                       classification_mapping_path: str,
                       images_directory_path: str
                       ) -> pd.DataFrame:
    """
    Get the classification filtered by existing images and one hot encoded with the corresponding mapping.
    :param classification_path: Path tot the classification file.
    :param classification_mapping_path: Path to the classification mapping.
    :param images_directory_path: Path to the images.
    :returns: The images' classification.
    """

    # Load the classification mapping.
    mapping = _load_classification_mapping(resolve_path(classification_mapping_path))

    # Get the image classification.
    # Load the classification.
    # Filter keep only the existing images.
    # Encode the classification into integers using the mappings.
    # One hot encode the classification.
    image_classification = _load_image_classification(resolve_path(classification_path)) \
        .pipe(
            _filter_keep_valid_images,
            resolve_path(images_directory_path),
        ).pipe(
            _encode_classification,
            mapping=mapping,
        ).pipe(
            _one_hot_encode_classification,
            mapping=mapping,
        )

    # Return the image classification.
    return image_classification
