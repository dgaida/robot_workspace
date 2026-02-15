from __future__ import annotations

# class defining a decorator to log functions entry and exit
# final
import inspect
import logging
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def log_start_end(verbose: bool = True) -> Callable[[F], F]:
    """
    Returns a decorator to log the start and end of a function.

    Args:
        verbose (bool): If True, logs will be printed.

    Returns:
        function: Decorator for logging start and end of functions.
    """

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger("robot_workspace")
            if verbose:
                logger.debug(f"START {func.__name__}")
            result = func(*args, **kwargs)
            if verbose:
                logger.debug(f"END {func.__name__}")
            return result

        return cast("F", wrapper)

    return decorator


def log_start_end_cls() -> Callable[[F], F]:
    """
    Decorator to log the start and end of a method, including class name and line number.
    """

    def decorator(func: F) -> F:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger("robot_workspace")
            # Get verbose attribute from the instance or class
            verbose = getattr(self, "_verbose", False)
            class_name = None
            func_line_number = None

            if verbose:
                # Retrieve class name
                class_name = self.__class__.__name__
                # Retrieve line number where the function is defined
                func_line_number = inspect.getsourcelines(func)[1]
                # Log start message
                logger.debug(f"START {func.__name__} (Class: {class_name}, Line: {func_line_number})")
            result = func(self, *args, **kwargs)
            if verbose:
                # Log end message
                logger.debug(f"END {func.__name__} (Class: {class_name}, Line: {func_line_number})")
            return result

        return cast("F", wrapper)

    return decorator
