# Konfiguration

Informationen zur Konfiguration des Workspace-Systems.

## Workspace-Konfiguration

Workspaces werden in der Workspace-Implementierung definiert:

```python
def _set_observation_pose(self):
    if self._id == "niryo_ws":
        self._observation_pose = PoseObjectPNP(
            x=0.173, y=-0.002, z=0.277,
            roll=-3.042, pitch=1.327, yaw=-3.027
        )
```

## Logging-Konfiguration

Das Paket verwendet das Standard-Python-Logging-Modul. Sie können den Log-Level anpassen:

```python
import logging
logging.getLogger("robot_workspace").setLevel(logging.DEBUG)
```
