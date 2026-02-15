# Robot Workspace

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/robot_workspace/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/robot_workspace)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Ein Python-Framework, das die Lücke zwischen Kamerabildern und physikalischer Robotermanipulation schließt. Es bietet die wesentlichen Datenstrukturen und Koordinatentransformationen, die benötigt werden, um erkannte Objekte von Visionssystemen in ausführbare Pick-and-Place-Ziele für Roboterarme umzuwandeln. Das Framework kümmert sich um Workspace-Kalibrierung, Objektrepräsentation mit physikalischen Eigenschaften und räumliches Denken – so können mit Kameras ausgestattete Roboter verstehen, "wo" sich Objekte befinden und "wie" sie in realen Koordinaten gegriffen werden können.

---

## 🎯 Überblick

Das `robot_workspace`-Paket bietet ein vollständiges Framework zur Verwaltung von Roboter-Workspaces, einschließlich:

- **🎯 Koordinatentransformationen**: Nahtlose Transformation zwischen Kamera- und Welt-Koordinatensystemen
- **📦 Objektrepräsentation**: Reichhaltige Objektmodelle mit Position, Dimensionen, Segmentierungsmasken und Orientierung
- **🗺️ Workspace-Management**: Definieren und Verwalten mehrerer Workspaces mit unterschiedlichen Konfigurationen
- **🔍 Räumliche Abfragen**: Finden von Objekten nach Position, Größe, Nähe oder benutzerdefinierten Kriterien
- **💾 Serialisierung**: JSON-basierte Serialisierung für Datenpersistenz und Kommunikation
- **🤖 Roboter-Unterstützung**: Native Unterstützung für Niryo Ned2 und WidowX 250 6DOF Roboter (Echt und Simulation)

---

## ✨ Hauptmerkmale

### Vision & Erkennung
- Integration der Objekterkennung mit Bounding Boxes, Segmentierungsmasken und physikalischen Eigenschaften
- Berechnung des Massenschwerpunkts und optimaler Greiferorientierungen
- Unterstützung für Multi-Objekt-Tracking und Management

### Koordinatensysteme
- Transformation zwischen relativen Bildkoordinaten (0-1) und Weltkoordinaten (Meter)
- Handhabung mehrerer Workspace-Konfigurationen mit unterschiedlichen Kameraposen
- Automatische Erkennung von Workspace-Grenzen

### Räumliches Denken
- Abfrage von Objekten nach räumlichen Beziehungen (links/rechts/oberhalb/unterhalb/nah bei)
- Finden des nächstgelegenen Objekts zu angegebenen Koordinaten
- Filtern nach Größe, Label oder benutzerdefinierten Kriterien

---


## 📦 Installation

```bash
pip install -e .
```

Für alle Features:
```bash
pip install -e ".[all]"
```

---

## 🚀 Schnellstart

```python
from robot_workspace import PoseObjectPNP, Object, Objects, NiryoWorkspaces

# 1. Arbeiten mit Poses
pose = PoseObjectPNP(x=0.2, y=0.1, z=0.05, roll=0.0, pitch=1.57, yaw=0.0)

# 2. Objektrepräsentation
obj = Object(
    label="pencil",
    u_min=100, v_min=100, u_max=200, v_max=200,
    mask_8u=None,
    workspace=workspace
)

# 3. Räumliche Abfragen
objects = Objects([obj1, obj2, obj3])
nearest, distance = objects.get_nearest_detected_object([0.25, 0.05])
```
