# Tests

Dokumentation der Teststrategie für `robot_workspace`.

## Teststruktur

Tests sind in zwei Kategorien unterteilt:

- **Unit-Tests**: Testen einzelne Klassen und Methoden isoliert (unter `tests/unit/`).
- **Integrationstests**: Testen das Zusammenspiel mehrerer Komponenten (unter `tests/integration/`).

## Tests ausführen

```bash
# Alle Tests ausführen
pytest

# Nur Unit-Tests
pytest tests/unit

# Mit Coverage-Bericht
pytest --cov=robot_workspace
```

## API-Dokumentationsprüfung

Wir verwenden `interrogate`, um sicherzustellen, dass alle öffentlichen APIs dokumentiert sind.

```bash
interrogate robot_workspace
```
