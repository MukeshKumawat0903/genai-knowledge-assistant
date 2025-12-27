"""
Centralized Logging

This module provides centralized logging functionality for the application.
It configures logging with appropriate formatters, handlers, and levels.

Features:
- Console and file logging
- Structured log formatting
- Log rotation
- Different log levels per module
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """
        Format log record with colors.
        
        Args:
            record: Log record
            
        Returns:
            Formatted string
        """
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        
        # Apply color to level name
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format with parent formatter
        return super().format(record)


class Logger:
    """
    Centralized logger configuration.
    """
    
    def __init__(
        self,
        name: str = "genai_assistant",
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_colors: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_colors: Enable colored console output
            
        TODO:
        - Create logger instance
        - Set logging level
        - Add console handler if enabled
        - Add file handler if enabled
        - Configure formatters
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_colors = enable_colors
        
        self.logger = None
    
    def setup(self) -> logging.Logger:
        """
        Set up and configure logger.
        
        Returns:
            Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add console handler if enabled
        if self.enable_console:
            self._add_console_handler(logger)
        
        # Add file handler if enabled
        if self.enable_file:
            self._add_file_handler(logger)
        
        self.logger = logger
        return logger
    
    def _add_console_handler(self, logger: logging.Logger) -> None:
        """
        Add console handler to logger.
        
        Args:
            logger: Logger instance
        """
        # Create StreamHandler for stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Create formatter (colored if enabled)
        formatter = self._get_console_formatter()
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    def _add_file_handler(self, logger: logging.Logger) -> None:
        """
        Add rotating file handler to logger.
        
        Args:
            logger: Logger instance
        """
        # Create logs directory if not exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        log_filename = self.log_dir / f"{self.name}.log"
        
        # Create RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.log_level)
        
        # Set formatter
        formatter = self._get_file_formatter()
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def _get_console_formatter(self) -> logging.Formatter:
        """
        Get console log formatter.
        
        Returns:
            Formatter instance
        """
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if self.enable_colors:
            return ColoredFormatter(format_string)
        else:
            return logging.Formatter(format_string)
    
    def _get_file_formatter(self) -> logging.Formatter:
        """
        Get file log formatter.
        
        Returns:
            Formatter instance
        """
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        return logging.Formatter(format_string)


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create logger instance.
    
    Args:
        name: Optional logger name (defaults to module name)
        
    Returns:
        Logger instance
        
    TODO:
    - If global logger not initialized, create it
    - If name provided, get child logger
    - Return logger
    """
    global _logger
    if _logger is None:
        logger_config = Logger()
        _logger = logger_config.setup()
    
    if name:
        return _logger.getChild(name)
    return _logger


def setup_logger(
    name: str = "genai_assistant",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    **kwargs
) -> logging.Logger:
    """
    Set up application logger.
    
    Args:
        name: Logger name
        log_dir: Log directory
        log_level: Logging level
        **kwargs: Additional configuration
        
    Returns:
        Configured logger
        
    TODO:
    - Create Logger instance
    - Set up with provided config
    - Store as global logger
    - Return logger
    """
    global _logger
    logger_config = Logger(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        **kwargs
    )
    _logger = logger_config.setup()
    return _logger


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
        
    TODO:
    - Create wrapper that logs:
      - Function name
      - Arguments
      - Return value
      - Execution time
      - Any exceptions
    - Return wrapper
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        # TODO: Implement logging logic
        return func(*args, **kwargs)
    return wrapper


def log_exception(exc_type, exc_value, exc_traceback):
    """
    Log uncaught exceptions.
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
        
    TODO:
    - Get logger
    - Log exception with full traceback
    - Format nicely for debugging
    """
    pass


# Set up exception logging
def setup_exception_logging():
    """
    Set up logging for uncaught exceptions.
    
    TODO:
    - Set sys.excepthook to log_exception
    """
    pass


class LogContext:
    """
    Context manager for temporary log level changes.
    """
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize context manager.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
            
        TODO:
        - Store logger and new level
        - Store original level
        """
        self.logger = logger
        self.level = getattr(logging, level.upper())
        self.original_level = None
    
    def __enter__(self):
        """
        Enter context - change log level.
        
        TODO:
        - Save original level
        - Set new level
        - Return self
        """
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context - restore log level.
        
        TODO:
        - Restore original level
        """
        pass
