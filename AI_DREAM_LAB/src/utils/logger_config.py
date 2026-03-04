
from loguru import logger
import sys

def setup_logging(log_file=None):
    """Configure logging settings."""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    
    # File logging if specified
    if log_file:
        logger.add(log_file, 
                  format="{time} {level} {message}",
                  rotation="500 MB",
                  level="DEBUG")
    
    return logger
