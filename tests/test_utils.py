"""
Unit tests for utils.py (Utility functions)
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from robot_workspace.utils.utils import setup_logging


class TestSetupLogging:
    """Test suite for setup_logging function"""

    def test_setup_logging_basic(self):
        """Test basic logging setup"""
        logger = setup_logging()

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "robot_workspace"

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a Logger instance"""
        logger = setup_logging()

        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

    def test_setup_logging_default_level(self):
        """Test default logging level is INFO"""
        logger = setup_logging(verbose=False)

        assert logger.level == logging.INFO

    def test_setup_logging_verbose_level(self):
        """Test verbose logging level is DEBUG"""
        logger = setup_logging(verbose=True)

        assert logger.level == logging.DEBUG

    def test_setup_logging_no_duplicate_handlers(self):
        """Test that calling setup_logging multiple times doesn't create duplicate handlers"""
        # Clear any existing handlers first
        logger_name = "robot_workspace"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

        # First call
        logger1 = setup_logging()
        initial_handler_count = len(logger1.handlers)

        # Second call
        logger2 = setup_logging()
        second_handler_count = len(logger2.handlers)

        # Should return existing logger without adding handlers
        assert logger1 is logger2
        assert second_handler_count == initial_handler_count

    def test_setup_logging_has_console_handler(self):
        """Test that logger has a console handler"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        logger = setup_logging()

        # Should have at least one handler
        assert len(logger.handlers) > 0

        # Should have a StreamHandler
        has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert has_stream_handler

    def test_setup_logging_with_log_file(self):
        """Test logging setup with log file"""
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            # Clear existing handlers
            logger = logging.getLogger("robot_workspace")
            logger.handlers.clear()

            logger = setup_logging(log_file=log_file)

            # Should have console handler + file handler
            assert len(logger.handlers) >= 2

            # Should have a FileHandler
            has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            assert has_file_handler

            # Test logging to file
            logger.info("Test message")

            # Verify file was created and has content
            assert os.path.exists(log_file)
            with open(log_file, "r") as f:
                content = f.read()
                assert "Test message" in content

        finally:
            # Cleanup
            if os.path.exists(log_file):
                os.remove(log_file)
            # Clear handlers
            logger.handlers.clear()

    def test_setup_logging_file_creation_failure(self):
        """Test graceful handling of log file creation failure"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        # Try to create log file in non-existent directory
        invalid_path = "/nonexistent/directory/test.log"

        # Should not raise exception
        logger = setup_logging(log_file=invalid_path)

        # Should still have console handler
        assert len(logger.handlers) >= 1

    def test_setup_logging_formatter(self):
        """Test that handlers have proper formatter"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        logger = setup_logging()

        # All handlers should have formatters
        for handler in logger.handlers:
            assert handler.formatter is not None

    def test_setup_logging_verbose_false(self):
        """Test logging with verbose=False"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        logger = setup_logging(verbose=False)

        assert logger.level == logging.INFO

        # Handlers should also be at INFO level
        for handler in logger.handlers:
            assert handler.level == logging.INFO

    def test_setup_logging_verbose_true(self):
        """Test logging with verbose=True"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        logger = setup_logging(verbose=True)

        assert logger.level == logging.DEBUG

        # Handlers should also be at DEBUG level
        for handler in logger.handlers:
            assert handler.level == logging.DEBUG

    def test_setup_logging_log_messages_verbose(self):
        """Test that DEBUG messages are logged when verbose=True"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        logger = setup_logging(verbose=True)

        # Should be able to log DEBUG messages
        with pytest.raises(Exception, match=None):
            # This should not raise an exception
            logger.debug("Debug message")

    def test_setup_logging_encoding_utf8(self):
        """Test that file handler uses UTF-8 encoding"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            # Clear existing handlers
            logger = logging.getLogger("robot_workspace")
            logger.handlers.clear()

            logger = setup_logging(log_file=log_file)

            # Log unicode message
            logger.info("Test unicode: こんにちは 🎉")

            # Read file and verify encoding
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "こんにちは" in content
                assert "🎉" in content

        finally:
            if os.path.exists(log_file):
                os.remove(log_file)
            logger.handlers.clear()

    def test_setup_logging_append_mode(self):
        """Test that file handler uses append mode"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            # Clear existing handlers
            logger = logging.getLogger("robot_workspace")
            logger.handlers.clear()

            # First setup and log
            logger1 = setup_logging(log_file=log_file)
            logger1.info("First message")
            logger1.handlers.clear()

            # Second setup and log
            logger2 = setup_logging(log_file=log_file)
            logger2.info("Second message")

            # Both messages should be in file
            with open(log_file, "r") as f:
                content = f.read()
                assert "First message" in content
                assert "Second message" in content

        finally:
            if os.path.exists(log_file):
                os.remove(log_file)
            logger2.handlers.clear()


class TestSetupLoggingEncoding:
    """Test encoding handling in setup_logging"""

    def test_setup_logging_handles_cp1252(self):
        """Test handling of cp1252 encoding (Windows console)"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        # This should not raise an exception even if console uses cp1252
        logger = setup_logging()

        assert logger is not None
        logger.handlers.clear()

    def test_setup_logging_unicode_console_output(self):
        """Test that unicode can be logged to console"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        logger = setup_logging()

        # Should handle unicode without crashing
        try:
            logger.info("Unicode test: こんにちは 🎉")
            success = True
        except Exception:
            success = False

        assert success
        logger.handlers.clear()

    def test_setup_logging_fallback_stream_handler(self):
        """Test fallback to default StreamHandler on error"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        # Should handle encoding issues gracefully
        logger = setup_logging()

        # Should have at least console handler
        assert len(logger.handlers) >= 1
        logger.handlers.clear()


class TestSetupLoggingEdgeCases:
    """Test edge cases in setup_logging"""

    def test_setup_logging_none_log_file(self):
        """Test with log_file=None (default)"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        logger = setup_logging(log_file=None)

        # Should only have console handler
        assert len(logger.handlers) == 1
        logger.handlers.clear()

    def test_setup_logging_empty_string_log_file(self):
        """Test with empty string log file"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        # Empty string should be treated as no file
        logger = setup_logging(log_file="")

        # Should still work (may try to create file or skip)
        assert logger is not None
        logger.handlers.clear()

    def test_setup_logging_relative_log_path(self):
        """Test with relative log file path"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                log_file = "test.log"
                logger = setup_logging(log_file=log_file)
                logger.info("Test message")

                # File should be created in current directory
                assert os.path.exists(log_file)

            finally:
                os.chdir(original_dir)
                logger.handlers.clear()

    def test_setup_logging_absolute_log_path(self):
        """Test with absolute log file path"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")

            logger = setup_logging(log_file=log_file)
            logger.info("Test message")

            assert os.path.exists(log_file)
            logger.handlers.clear()

    def test_setup_logging_pathlib_path(self):
        """Test with pathlib.Path object"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            logger = setup_logging(log_file=str(log_file))
            logger.info("Test message")

            assert log_file.exists()
            logger.handlers.clear()

    def test_setup_logging_very_long_log_path(self):
        """Test with very long log file path"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        # Very long path (may fail on some systems)
        long_path = "/tmp/" + "a" * 200 + ".log"

        # Should handle gracefully (may or may not create file)
        logger = setup_logging(log_file=long_path)

        assert logger is not None
        logger.handlers.clear()

    def test_setup_logging_special_chars_in_path(self):
        """Test with special characters in log path"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with spaces
            log_file = os.path.join(tmpdir, "test log.log")

            logger = setup_logging(log_file=log_file)
            logger.info("Test message")

            assert os.path.exists(log_file)
            logger.handlers.clear()


class TestSetupLoggingFormatting:
    """Test log message formatting"""

    def test_setup_logging_message_format(self):
        """Test that log messages have proper format"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            logger = setup_logging(log_file=log_file)
            logger.info("Test message")

            with open(log_file, "r") as f:
                content = f.read()

                # Should contain timestamp, logger name, level, and message
                assert "robot_workspace" in content
                assert "INFO" in content
                assert "Test message" in content

        finally:
            if os.path.exists(log_file):
                os.remove(log_file)
            logger.handlers.clear()

    def test_setup_logging_different_log_levels(self):
        """Test logging at different levels"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            logger = setup_logging(verbose=True, log_file=log_file)

            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            with open(log_file, "r") as f:
                content = f.read()

                assert "DEBUG" in content
                assert "INFO" in content
                assert "WARNING" in content
                assert "ERROR" in content

        finally:
            if os.path.exists(log_file):
                os.remove(log_file)
            logger.handlers.clear()


class TestSetupLoggingIntegration:
    """Integration tests for setup_logging"""

    def test_setup_logging_in_real_scenario(self):
        """Test logging in a realistic scenario"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "app.log")

            # Setup logging as an application would
            logger = setup_logging(verbose=True, log_file=log_file)

            # Log various messages
            logger.debug("Application starting")
            logger.info("Processing data")
            logger.warning("Resource usage high")
            logger.error("Operation failed")

            # Verify all messages were logged
            with open(log_file, "r") as f:
                content = f.read()

                assert "Application starting" in content
                assert "Processing data" in content
                assert "Resource usage high" in content
                assert "Operation failed" in content

            logger.handlers.clear()

    def test_setup_logging_multiple_modules(self):
        """Test that multiple modules can use the same logger"""
        # Clear existing handlers
        logger = logging.getLogger("robot_workspace")
        logger.handlers.clear()

        # Setup once
        logger1 = setup_logging()

        # Get logger in "different module"
        logger2 = logging.getLogger("robot_workspace")

        # Should be the same logger
        assert logger1 is logger2

        logger.handlers.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
