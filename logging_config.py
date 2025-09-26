"""
Centralized logging configuration for Conv2d project.

This module provides consistent logging setup across all components.
"""

import logging
import logging.config
import sys
from pathlib import Path


def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the Conv2d project.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'models': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'preprocessing': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'scripts': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'tests': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'detailed',
            'filename': str(log_file),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
        # Add file handler to all loggers
        for logger in config['loggers'].values():
            logger['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging configured for Conv2d project")
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Logging levels for easy access
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG


if __name__ == "__main__":
    # Example usage
    setup_logging(level=logging.DEBUG, log_file="logs/conv2d.log")
    
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")