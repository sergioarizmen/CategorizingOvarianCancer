import os
import sys
from .exception_decorator import log_exception


@log_exception
def resolve_path(path: str) -> str:
    """
    Resolve the path to a dependency file/directory or a file/directory within the currently working directory.
    If a dependency file/directory and a file/directory in the currently working directory have the same name,
        the dependency file/directory will take priority.
    In frozen mode (executable), files/directories will be first looked for in the MEIPASS file path.
    If not in frozen mode (develop mode), all files/directories will be looked for in the currently working directory.
    :param path: A relative path to the file/directory.
    :return: The absolute path file/directory.
    """

    # If the path is an absolute path.
    if os.path.isabs(path):
        # Return the path as is.
        return path

    # Check for a valid dependency path.
    # If in frozen instance (executable).
    if getattr(sys, 'frozen', False):
        # Get the absolute path, considering the MEIPASS directory as root.
        temp_path = os.path.join(sys._MEIPASS, path)

        # If the file exists at the MEIPASS directory.
        if os.path.exists(temp_path):
            # Return the dependency file path.
            return temp_path

    # Check for a valid path in the current working directory.
    # If not in frozen mode (develop mode).
    else:
        # Get the absolute path, considering current working directory as root.
        temp_path = os.path.join(os.getcwd(), path)

        # Return the currently working directory file path.
        return temp_path
