import numpy as np
import cv2 as cv


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
    patches = np.array([], dtype=image.dtype)

    # Loop through the image with a step size of 250 pixels.
    for y in range(0, height, 250):
        for x in range(0, width, 250):
            # Get the height and width of the current patch
            patch_height = min(250, height - y)
            patch_width = min(250, width - x)

            # Crop a h by 250 pixel patch from the image.
            patch = image[y:y + patch_height, x:x + patch_width]

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
                noise = np.random.randint(0, 256, (output_patch_height, output_patch_width, 3), dtype=patch.dtype)

                # Replace a patch sized space in the noise with zeroes.
                noise[0:patch_height, 0:patch_width, :] = patch

                # Replace the patch with the noise filled patch.
                patch = noise

            # Add the patch to the array of patches.
            np.append(patches, patch)

    # Return the patches.
    return patches


def preprocess_image(image):
    """
    Convert to grayscale and pass into patches.
    :param image: The image to preprocess.
    :return: The preprocessed patches.
    """

    # Convert the image to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Get the patched up image.
    image = _get_patches(image)

    # Return the preprocessed image.
    return image
