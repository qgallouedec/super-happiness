import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the given name and level.

    Args:
        name (`str`):
            The name of the logger.
        level (`int`, *optional*, defaults to `logging.INFO`):
            The logging level.

    Returns:
        `logging.Logger`:
            The configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
