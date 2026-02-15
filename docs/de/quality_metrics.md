# Qualitätsmetriken

Diese Seite bietet einen Überblick über die Qualitätsmetriken des Projekts, einschließlich Dokumentationsabdeckung und Testergebnisse.

## 📊 API-Dokumentationsabdeckung

Wir verwenden `interrogate`, um unsere API-Dokumentationsabdeckung zu messen. Unser Ziel ist **>95%**.

| Metrik | Status |
|--------|--------|
| Öffentliche API-Abdeckung | ![Interrogate Badge](../interrogate_badge.svg) |
| Docstring-Stil | Google |

## 🧪 Testabdeckung

| Kategorie | Abdeckung |
|-----------|-----------|
| Gesamt | [![codecov](https://codecov.io/gh/dgaida/robot_workspace/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/robot_workspace) |
| Unit-Tests | >90% |
| Integrationstests | Verifiziert |

## 🛠️ Code-Qualität

- **Linter**: Ruff
- **Formatierer**: Black
- **Typprüfung**: Mypy (Strict)
