from __future__ import annotations

"""
Utility functions for the robot_workspace package.
Contains helper functions for logging
"""

import logging


# Logging setup
def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the vision detection system.
    Handles Unicode output gracefully on Windows consoles.
    """
    import sys

    logger = logging.getLogger("robot_workspace")

    # FIXED: Check if logger already exists AND update its level
    if logger.handlers:
        print("utils.py: using an already existent logger")
        # Update the level even if logger exists
        level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(level)
        # Update handler levels too
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ---- Handle console encoding problems (Windows cp1252 etc.) ----
    try:
        stream = sys.stdout
        encoding = getattr(stream, "encoding", None)

        # If Windows console with cp1252, wrap stream to use UTF-8 with replacement
        if encoding is None or encoding.lower() in ["cp1252", "ansi_x3.4-1968"]:
            import io

            stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

        console_handler = logging.StreamHandler(stream)
    except Exception:
        # Fallback to default stream handler
        console_handler = logging.StreamHandler()

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # ---- Optional file logging ----
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file {log_file}: {e}")

    return logger
