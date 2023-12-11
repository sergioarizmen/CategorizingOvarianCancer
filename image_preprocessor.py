import cv2 as cv
import numpy as np

from default.exception_decorator import log_exception


@log_exception
def _get_patches(image,
                 output_patch_width: int = 250,
                 output_patch_height: int = 250,
                 threshold: float = 110.0,
                 ):
    """
    Make patches out of an image.
    :param image: The image to make patches out of.
    :param output_patch_width: Output patch width.
    :param output_patch_height: Output patch height.
    :param threshold: Threshold to remove or keep parts of the image.
    :returns: A numpy array of patches.
    """

    # Get the height and width of the image.
    height, width = image.shape[:2]

    # Initialize an empty array of patches.
    patches = []

    # Loop through the image with a step size of 250 pixels.
    for y in range(0, height, output_patch_height):
        for x in range(0, width, output_patch_width):
            # Get the height and width of the current patch
            patch_height = min(output_patch_height, height - y)
            patch_width = min(output_patch_width, width - x)

            # Crop a h by 250 pixel patch from the image.
            patch = image[y:y + patch_height, x:x + patch_width, ...]

            # Calculate the mean value of the patch pixels
            mean = np.mean(patch)

            # If the mean value is above the threshold, save the patch as a new image.
            if abs(mean - 128) > threshold:
                # Skip the current patch.
                continue

            # If the patch is smaller than the required size in pixels.
            if patch_height < output_patch_height or patch_width < output_patch_width:
                # Pad the image with random noise.
                # Generate random noise with the same shape and type as the patch.
                noise = np.random.randint(0, 256, (output_patch_height, output_patch_width), dtype=patch.dtype)

                # Replace a patch sized space in the noise with zeroes.
                noise[0:patch_height, 0:patch_width, ...] = patch

                # Replace the patch with the noise filled patch.
                patch = noise

            # Add the patch to the array of patches.
            patches.append(patch)

    # Return the patches.
    return np.array(patches, dtype=image.dtype)


@log_exception
def preprocess_image(image):
    """
    Convert to grayscale and pass into patches of type 'float32'.
    :param image: The image to preprocess.
    :return: The preprocessed patches.
    """

    # Convert the image to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Get the patched up image.
    image = _get_patches(image)

    # Transform the number type.
    image = image.astype('float32')

    # Return the preprocessed image.
    return image
