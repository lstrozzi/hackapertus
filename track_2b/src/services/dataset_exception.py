"""
Dataset Exception Module

This module provides exceptions for dataset service operations.
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


class DatasetReadOnlyError(PermissionError):
    """Raised when a write is attempted on a read-only dataset."""


log_end_file(__file__)
