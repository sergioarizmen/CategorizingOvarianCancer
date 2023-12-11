import logging


class DefaultFormatter(logging.Formatter):
    """
    Defines default logging formatter.
    """

    # Set default formatting colors.
    grey = "\x1b[37m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Set default format.
    format = "%(asctime)s %(levelname)s %(message)s"

    # For each default logging level set the formatting string with the corresponding colors.
    # For console processing: first set the color, then send the message, finally reset the color.
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Formatting function called on all messages sent to the console.
        :param record:
        :return:
        """

        # Get the corresponding record level's format.
        log_format = self.FORMATS.get(record.levelno)

        # Return the formatted record.
        return logging.Formatter(log_format).format(record)
