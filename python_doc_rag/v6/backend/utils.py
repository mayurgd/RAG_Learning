import time
import functools

from v6.logger import loggers_utils

logger = loggers_utils(__name__)


def time_it(func):
    """
    A decorator to measure and log the execution time of a function in HH:MM:SS.ss format.

    Args:
        func (Callable): The function to be timed.

    Returns:
        Callable: A wrapped function that logs execution time.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time

        # Convert runtime to HH:MM:SS.ss format
        hours, rem = divmod(runtime, 3600)
        minutes, seconds = divmod(rem, 60)
        formatted_time = f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"

        logger.info(f"Function '{func.__name__}' executed in {formatted_time}")
        return result

    return wrapper
