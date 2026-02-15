# Erste Schritte

Diese Anleitung hilft Ihnen beim Einstieg in das `robot_workspace`-Paket.

## Kernkonzepte

Das Repository bietet drei Hauptfunktionen:

1. **Pose-Repräsentation** - 6-DOF Posen (Position + Orientierung) für Objekte und Roboterziele.
2. **Objekt-Management** - Erkannte Objekte mit physikalischen Eigenschaften, Positionen und räumlichen Abfragen.
3. **Koordinatentransformation** - Konvertierung von Kamerakoordinaten in Roboter-Weltkoordinaten.

## Grundlegende Verwendung

### 1. Arbeiten mit Poses (Position + Orientierung)

```python
from robot_workspace import PoseObjectPNP

# Pose erstellen
pose = PoseObjectPNP(x=0.2, y=0.1, z=0.05, roll=0.0, pitch=1.57, yaw=0.0)
print(f"Position: [{pose.x}, {pose.y}, {pose.z}]")

# Offsets hinzufügen
pick_pose = pose.copy_with_offsets(z_offset=-0.02)  # Greifer 2cm absenken
```

### 2. Repräsentation erkannter Objekte

```python
from robot_workspace import Object

obj = Object(
    label="pencil",
    u_min=100, v_min=100, u_max=200, v_max=200,
    mask_8u=None,
    workspace=workspace
)

print(f"Objekt '{obj.label()}' bei [{obj.x_com():.2f}, {obj.y_com():.2f}] Metern")
```

### 3. Räumliche Abfragen

```python
from robot_workspace import Objects, Location

objects = Objects([obj1, obj2, obj3])

# Nach räumlichen Beziehungen suchen
left_objects = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.2, 0.0]
)

# Nächstgelegenes Objekt finden
nearest, distance = objects.get_nearest_detected_object([0.25, 0.05])
```
