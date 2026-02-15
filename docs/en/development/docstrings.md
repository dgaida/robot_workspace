# Docstring Style Guide

We use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all docstrings in this project.

## Format

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short description of the function.

    Longer description explaining the logic, side effects, or
    anything else that might be important for the user.

    Args:
        param1 (int): Description of the first parameter.
        param2 (str): Description of the second parameter.

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: Description of when this error is raised.
    """
```

## Classes

```python
class MyClass:
    """
    Summary of the class.

    Attributes:
        attr1 (int): Description of attr1.
        attr2 (str): Description of attr2.
    """
```
