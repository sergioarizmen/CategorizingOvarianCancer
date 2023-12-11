import logging
import os
import sys

from default.log_formatter import DefaultFormatter


def set_default_logging_configuration():
    """Set default logging configuration. With the default logging formatter."""

    # If in frozen instance (executable).
    if getattr(sys, 'frozen', False):
        # Do an empty system call to refresh the console configuration.
        # This fixes a bug in frozen instances where the color will not display properly.
        os.system("")

    # Set logging level to ERROR.
    logging.basicConfig(
        level=logging.ERROR,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set the formatter to the DefaultFormatter.
    set_default_formatter()


def set_default_formatter():
    """Set the logging formatter to the DefaultFormatter."""

    # Get the root logger object.
    root_logger = logging.getLogger()

    # Empty the root logger handlers.
    root_logger.handlers = []

    # Get a logging stream handler.
    logger_stream_handler = logging.StreamHandler()

    # Set the stream handler's formatter to the default formatter.
    logger_stream_handler.setFormatter(DefaultFormatter())

    # Add the logging stream handler to the logging handlers.
    logging.root.addHandler(logger_stream_handler)
