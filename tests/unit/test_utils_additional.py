"""
Additional unit tests for utils to increase coverage.
"""

import logging
import sys
from unittest.mock import MagicMock, patch
import pytest
from robot_workspace.utils.utils import setup_logging


def test_setup_logging_windows_encoding():
    """Test setup_logging with simulated Windows/CP1252 encoding."""
    mock_stdout = MagicMock()
    mock_stdout.encoding = "cp1252"

    # We need to mock sys.stdout.buffer as well because io.TextIOWrapper uses it
    mock_stdout.buffer = MagicMock()

    with patch("sys.stdout", mock_stdout), \
         patch("io.TextIOWrapper") as mock_wrapper:
        # Reset the logger to ensure we hit the initialization code
        logger = logging.getLogger("robot_workspace")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(verbose=True)

        # Verify TextIOWrapper was called due to cp1252 encoding
        mock_wrapper.assert_called()


def test_setup_logging_exception_in_stream_setup():
    """Test setup_logging when an exception occurs during stream setup."""
    mock_stdout = MagicMock()
    # Trigger an exception when accessing encoding
    type(mock_stdout).encoding = property(lambda x: exec('raise Exception("Test Exception")'))

    with patch("sys.stdout", mock_stdout):
        # Reset the logger
        logger = logging.getLogger("robot_workspace")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # This should hit the 'except Exception' block
        logger = setup_logging()
        assert len(logger.handlers) > 0


def test_setup_logging_file_error(caplog):
    """Test setup_logging when file logging fails."""
    # Using a path that should be invalid or unwritable
    invalid_path = "/non_existent_directory_jules/test.log"

    # Reset the logger
    logger = logging.getLogger("robot_workspace")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    with caplog.at_level(logging.WARNING):
        setup_logging(log_file=invalid_path)

    assert f"Could not create log file {invalid_path}" in caplog.text


def test_setup_logging_already_initialized():
    """Test setup_logging when logger already has handlers."""
    logger = logging.getLogger("robot_workspace")
    if not logger.handlers:
        setup_logging()

    initial_level = logger.level

    # Call again with different verbosity
    setup_logging(verbose=True)
    assert logger.level == logging.DEBUG

    setup_logging(verbose=False)
    assert logger.level == logging.INFO
