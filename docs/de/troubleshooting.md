# Fehlerbehebung

Häufige Probleme und deren Lösungen.

## Objekt nicht gefunden

Wenn `get_detected_object` den Wert `None` zurückgibt:
- Überprüfen Sie, ob die Koordinaten innerhalb der Workspace-Grenzen liegen.
- Stellen Sie sicher, dass das Label korrekt geschrieben ist.

## Koordinatentransformation ungenau

- Überprüfen Sie die Kamera-Kalibrierung.
- Stellen Sie sicher, dass die Workspace-Ecken korrekt definiert sind.
- Prüfen Sie, ob die Bildauflösung mit der Konfiguration übereinstimmt.
