import logging
from functools import wraps


def log_exception(func):
    """
    Wrapper for logging all exceptions raised and not handled by functions.
    :param func: Function to wrap.
    :return: Wrapped function.
    """

    # Create the wrapper for the function.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to run the wrapped function as normal.
        try:
            # Return the wrapped function results.
            return func(*args, **kwargs)

        # Handle all exceptions.
        except Exception as err:
            # Print the error to the console and the function of origin.
            logging.error(f"{err} at function {func.__qualname__}.")
            # Raise the exception back.
            raise

    # Return the wrapper to the decorator.
    return wrapper
