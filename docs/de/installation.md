# Installation

Anleitung zur Installation des `robot_workspace`-Pakets.

## Voraussetzungen

- Python 3.9 oder höher
- pip Paketmanager

## Basis-Installation

```bash
# Repository klonen
git clone https://github.com/dgaida/robot_workspace.git
cd robot_workspace

# Im Entwicklungsmodus installieren
pip install -e .
```

## Mit Roboter-Unterstützung

```bash
# Niryo Ned2 Unterstützung
pip install -e ".[niryo]"

# Alle Features
pip install -e ".[all]"
```

## Entwickler-Installation

```bash
pip install -e ".[dev]"
```

Dies installiert zusätzliche Werkzeuge:
- `pytest` und `pytest-cov` für Tests
- `black` für Code-Formatierung
- `ruff` für Linting
- `mypy` für Typprüfung
- `pre-commit` Hooks
