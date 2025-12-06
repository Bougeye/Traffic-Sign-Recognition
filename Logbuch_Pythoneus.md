# Task 1: Data Loading Pipeline


## Overview
In diesem Schritt wurde eine vollständige Datenlade-Pipeline für das GTSRB-Dataset implementiert.
Die Pipeline lädt:
-   Bilder
-   Labels
-   Concept-Vektoren (binäre Feature-Annotationen)

Der Dataset-Output entspricht exakt der Vorgabe:
    (image_tensor, (concept_vector, label))


## Projektstruktur (relevante Teile)
    config/dataset.yml
    data/GTSRB/Final_Training/Images/
    data/GTSRB/concepts_per_class.csv
    src/data/gtsrb_dataset.py
    scripts/test_dataset.py


## Config – dataset.yml
    dataset:
    root_dir: "data/GTSRB/Final_Training/Images"
    concept_csv: "data/GTSRB/concepts_per_class.csv"
    image_size: 64

Pfade und Einstellungen werden nicht im Code hardcodiert, sondern über YAML verwaltet.


## Custom Dataset – gtsrb_dataset.py
Der Dataset:
-   lädt alle Bilder aus der Ordnerstruktur
-   liest die Concept-CSV
-   ignoriert class_name
-   erzeugt für jede Klasse einen binären Concept-Vektor
-   gibt (img, (concept_vector, label)) zurück


## Test – test_dataset.py
Ausführung:
    python scripts/test_dataset.py

Erwartete Ausgabe:
    torch.Size([8, 3, 64, 64])   # Bilder
    torch.Size([8, 43])          # Concept-Vektoren
    torch.Size([8])              # Labels

Damit ist bestätigt, dass:
-   Daten korrekt geladen werden
-   Konzepte richtig gemappt sind
-   Die Pipeline PyTorch-konform arbeitet


## Task 1 – abgeschlossen
-   YAML-Config
-   Custom Dataset
-   Concept-Integration
-   Funktionierender DataLoader
