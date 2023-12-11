import os

import numpy as np
import tensorflow as tf
from joblib import dump, load
from sklearn.neural_network import MLPClassifier

from cnn_model import CNNModel
from default.exception_decorator import log_exception
from default.resolve_path import resolve_path
from image_preprocessor import preprocess_image


class CNNBoostModel:
    """Use multiple classification CNN models and a GD classifier model to resolve the image type."""

    @log_exception
    def __init__(self,
                 classes: int,
                 input_width: int,
                 input_height: int,
                 with_rotations: bool = True,
                 epochs: int = 10,
                 ):
        """
        :param classes: Number of classes the model should handle.
        :param input_width: Input image size.
        :param input_height: Input image width.
        :param epochs: Number of trains on each image.
        """

        # Initialize parameters.
        self._classifier_model = None
        self._cnn_models = []
        self._classes = classes
        self._input_width = input_width
        self._input_height = input_height
        self._epochs = epochs
        self._with_rotations = with_rotations

    @log_exception
    def initialize(self):
        """
        Initialize the convolutional neural networks, adding a CNN with two classes for each assigned class.
        And, initialize an SGD classifier for each assigned class.
        """

        # If the classifiers model has not been initialized.
        if not self._classifier_model:
            self._classifier_model = MLPClassifier(max_iter=1000, tol=1e-3)

        # If the CNN's have not been initialized.
        if not self._cnn_models:
            # Iterate once per class.
            for _ in range(self._classes):
                # Initialize a sub model for the class estimator.
                sub_model = CNNModel(
                    classes=2,
                    input_width=self._input_width,
                    input_height=self._input_height,
                    with_rotations=self._with_rotations,
                    epochs=self._epochs,
                )

                # Initialize the sub model.
                sub_model.initialize()

                # Add the sub model to the sub models' array.
                self._cnn_models.append(sub_model)

    @log_exception
    def save(self, path: str):
        """
        Save the model to the given path.
        :param path: The path to save the model to.
        """

        # Get the resolved path.
        path = resolve_path(path)

        # Iterate over all cnn models, indexed.
        for i, cnn_model in enumerate(self._cnn_models):
            # Save the cnn model to a directory in the resolved path.
            cnn_model.save(os.path.join(path, f'cnn_{i}'))

        # Save the classifier in the resolved path.
        dump(self._classifier_model, os.path.join(path, f'classifier.joblib'))

    @log_exception
    def load(self, path: str):
        """
        Load the model from a given path.
        :param path: Path to load the model from.
        """

        # Get the resolved path.
        path = resolve_path(path)

        # Iterate over all subdirectories by name.
        for name in os.listdir(path):
            # If the name starts with 'cnn_'.
            if name.startswith("cnn_"):
                # Initialize a sub model for the class estimator.
                sub_model = CNNModel(
                    classes=2,
                    input_width=self._input_width,
                    input_height=self._input_height,
                    with_rotations=self._with_rotations,
                    epochs=self._epochs,
                )

                # Load the model from the sub path.
                sub_model.load(os.path.join(path, name))

                # Add the sub model to the sub models' array.
                self._cnn_models.append(sub_model)

        # Load the classifier model.
        self._classifier_model = load(os.path.join(path, 'classifier.joblib'))

    @log_exception
    def train(self, image, classification):
        """
        Train based on a single image.
        :param image: Image to train on.
        :param classification: Array of classifications corresponding to the image.
        """

        # Break the image into smaller patches.
        patches = preprocess_image(image)

        # Get the corresponding classification for each patch.
        # First, repeat the classification for the hole image by the number of patches.
        # Then, reshape the resulting array back into the correct classification shape.
        patch_classification = np.tile(
            classification,
            patches.shape[0],
        ).reshape(
            patches.shape[0],
            classification.size,
        )

        # If the number of classification columns doesn't match the number of classes.
        if classification.size != self._classes:
            # Raise an value error.
            raise ValueError(f'Number of classifications doesn\'t ({classification.size}) '
                             f'match the number of specified classes (f{self._classes}).')

        # Initialize an empty list of predictions.
        predictions = []

        # Iterate over the classifications.
        for i in range(self._classes):
            # Transform the classification into bi-column classifications.
            classification = tf.keras.utils.to_categorical(patch_classification[:, i], num_classes=2)

            # Train the corresponding sub model with the classification.
            self._cnn_models[i].train(patches, classification)

            # Get the final training results.
            prediction = np.sum(self._cnn_models[i].predict(patches), axis=0) / patches.shape[0]

            # Append the prediction.
            predictions.append(prediction)

        # Train the classifier model.
        self._classifier_model.partial_fit(
            np.array(predictions).reshape(1, -1),
            [patch_classification[0, :]],
            classes=[i for i in range(self._classes)],
        )

    @log_exception
    def predict(self, image):
        """
        Predict the label of a given images.
        :param image: Image to predict on.
        :return: Return the prediction labels tested with the model.
        """

        # Break the image into smaller patches.
        patches = preprocess_image(image)

        # Initialize an empty list of predictions.
        predictions = []

        # Iterate over the classifications.
        for i in range(self._classes):
            # Get the final training results.
            prediction = np.sum(self._cnn_models[i].predict(patches), axis=0) / patches.shape[0]

            # Append the prediction.
            predictions.append(prediction)

        # Return the classifier prediction.
        return self._classifier_model.predict(predictions)
