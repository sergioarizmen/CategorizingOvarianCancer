import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses


class CNNModel:
    """Convolutional Neural Network model."""

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

        # Configure the CNN.
        self._configure_cnn()

    def _configure_cnn(self):
        """Configure the convolutional neural network."""

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
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train(self, images, classification):
        """
        Train based on a set of images.
        :param images: Array-like of images.
        :param classification: Array of classifications corresponding to each image.
        """

        # Run a train step on the model.
        self._model.fit(images, classification, epochs=self._epochs)
