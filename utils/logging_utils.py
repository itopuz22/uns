"""Logging utilities for Industrial Quality Control System."""

import os
import sys
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


class AuditLogger:
    """Logger for classification audit trail."""

    def __init__(self, filepath: str, fields: list):
        self.filepath = Path(filepath)
        self.fields = fields
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create audit log file with headers if it doesn't exist."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.fields)

    def log(self, **kwargs) -> None:
        """Log a classification event."""
        row = [kwargs.get(field, '') for field in self.fields]
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def setup_logging(
    name: str = 'qc_system',
    log_level: str = 'INFO',
    log_dir: Optional[str] = None,
    max_size_mb: int = 10,
    backup_count: int = 5,
    colorized: bool = True
) -> logging.Logger:
    """
    Set up logging with file and console handlers.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        max_size_mb: Maximum log file size in MB
        backup_count: Number of backup files to keep
        colorized: Enable colored console output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if colorized:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path / f'{name}.log',
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'qc_system') -> logging.Logger:
    """Get an existing logger or create a basic one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Create basic console handler if no handlers exist
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self) -> 'PerformanceTimer':
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        import time
        self.end_time = time.perf_counter()
        self.logger.debug(f"{self.name}: {self.elapsed_ms:.2f}ms")

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
