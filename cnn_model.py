from default.resolve_path import resolve_path
from default.exception_decorator import log_exception
from tensorflow.python.keras import layers, models, losses


class CNNModel:
    """Convolutional Neural Network model."""

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
        self._model = None
        self._classes = classes
        self._input_width = input_width
        self._input_height = input_height
        self._epochs = epochs
        self._with_rotations = with_rotations

    @log_exception
    def initialize(self):
        """Initialize the convolutional neural network."""

        # Initialize a sequential NN model.
        self._model = models.Sequential()

        # Configure input layer.
        self._model.add(layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(self._input_height, self._input_width, 1)
        ))

        # Configure model layers.
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(128, activation='linear'))

        # Configure output layers.
        self._model.add(layers.Dense(self._classes, activation='softmax'))

        # Compile model shape.
        self._model.compile(
            optimizer='adam',
            loss=losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    @log_exception
    def save(self, path: str):
        """
        Save the model to the given path.
        :param path: The path to save the model to.
        """

        # Save the model to the resolved path.
        self._model.save(resolve_path(path))

    @log_exception
    def load(self, path: str):
        """
        Load the model from a given path.
        :param path: Path to load the model from.
        """

        # Get the model from the resolved path.
        self._model = models.load_model(resolve_path(path))

    @log_exception
    def train(self, images, classification):
        """
        Train based on a set of images.
        :param images: Array-like of images.
        :param classification: Array of classifications corresponding to each image.
        """

        # Run a train step on the model.
        self._model.fit(images, classification, epochs=self._epochs)

    @log_exception
    def predict(self, images):
        """
        Predict the label of a given images.
        :param images: Image to predict on.
        :return: Return the prediction labels tested with the model.
        """

        # Return the prediction images.
        return self._model.predict(images)
