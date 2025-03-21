import logging
from typing import Optional
from contextvars import ContextVar


def loggers_utils(log_name: str) -> logging.Logger:
    """Instance of Python Logger Class for local logging.

    Args:
        log_name (str): Name of the logger instance.

    Returns:
        logging.Logger: Instance of the logger.
    """
    logger = logging.getLogger(log_name)
    return logger


class AppFilter(logging.Filter):
    def __init__(self, correlation_id_ctx_var, name: str = "") -> None:
        """
        Initialize the AppFilter with a correlation ID context variable.

        Args:
            correlation_id_ctx_var (ContextVar): A context variable that stores the correlation ID.
            name (str): Optional. The name of the filter. Defaults to an empty string.
        """
        self._correlation_id_ctx_var = correlation_id_ctx_var
        super().__init__(name)

    def filter(self, record):
        """
        Modifies the logging record to include a correlation ID.
        """
        record.correlation_id = self._correlation_id_ctx_var.get()
        return True


class CustomHandler(logging.Handler):
    """
    A custom logging handler that processes multiline string records into separate log entries.
    """

    def __init__(self, stream=None):
        super(CustomHandler, self).__init__()
        self.stream = stream

    def emit(self, record):
        try:
            messages = record.msg.split("\n")
            for message in messages:
                temp_record = logging.makeLogRecord(record.__dict__)
                temp_record.msg = message
                self.stream.write(self.format(temp_record) + "\n")
                self.stream.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    correlation_id_ctx_var: ContextVar,
    log_to_file: bool = False,
    log_file: Optional[str] = "app.log",
):
    """
    Sets up local logging configuration with correlation ID support.

    Args:
        correlation_id_ctx_var (ContextVar): The context variable storing the correlation ID.
        log_to_file (bool): Whether to save logs to a file. Defaults to False.
        log_file (Optional[str]): The file path for saving logs. Defaults to "app.log".
    """
    formatter = logging.Formatter(
        "%(levelname)s - %(name)s - [%(correlation_id)s] - %(message)s"
    )

    handlers = []

    # Console Handler
    stream_handler = CustomHandler(stream=logging.StreamHandler().stream)
    stream_handler.addFilter(AppFilter(correlation_id_ctx_var))
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    # File Handler (if enabled)
    if log_to_file and log_file:
        file_stream = open(log_file, "a")
        file_handler = CustomHandler(stream=file_stream)
        file_handler.addFilter(AppFilter(correlation_id_ctx_var))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
