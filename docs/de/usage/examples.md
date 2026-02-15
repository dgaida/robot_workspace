# Beispiele

Hier finden Sie praktische Beispiele für die Verwendung des `robot_workspace`-Pakets.

## Kompletter Pick-and-Place Workflow

```python
from robot_workspace import Objects, Object, NiryoWorkspaces

# 1. Workspaces initialisieren
workspaces = NiryoWorkspaces(env)
ws = workspaces.get_home_workspace()

# 2. Erkannte Objekte hinzufügen
detected = [
    {"label": "Würfel", "bbox": [100, 100, 150, 150]},
    {"label": "Zylinder", "bbox": [300, 200, 350, 250]}
]

objs = Objects()
for item in detected:
    objs.append(Object(
        item["label"],
        item["bbox"][0], item["bbox"][1],
        item["bbox"][2], item["bbox"][3],
        None, ws
    ))

# 3. Zielobjekt finden
target = objs.get_largest_detected_object()[0]

# 4. Greifpose abrufen
pick_pose = target.pose_com()
```

## Serialisierung

```python
# In JSON konvertieren
json_data = target.to_json()

# Aus Dictionary wiederherstellen
new_obj = Object.from_dict(data_dict, ws)
```
