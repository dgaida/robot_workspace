# Docstring-Styleguide

Wir verwenden den [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) für alle Docstrings in diesem Projekt.

## Format

```python
def funktions_name(param1: int, param2: str) -> bool:
    """
    Kurze Beschreibung der Funktion.

    Längere Beschreibung, die die Logik, Seiteneffekte oder
    alles andere erklärt, was für den Benutzer wichtig sein könnte.

    Args:
        param1 (int): Beschreibung des ersten Parameters.
        param2 (str): Beschreibung des zweiten Parameters.

    Returns:
        bool: Beschreibung des Rückgabewerts.

    Raises:
        ValueError: Beschreibung, wann dieser Fehler ausgelöst wird.
    """
```

## Klassen

```python
class MeineKlasse:
    """
    Zusammenfassung der Klasse.

    Attributes:
        attr1 (int): Beschreibung von attr1.
        attr2 (str): Beschreibung von attr2.
    """
```
