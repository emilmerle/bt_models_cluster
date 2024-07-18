## Vorraussetzungen
- Python vorhanden (min. Version 3.9?)
- (torch und torchvision vorhanden) auch in requirements.txt
- unzip vorhanden
- wget vorhanden

## Um Auszuführen
- git Repository klonen (https://github.com/emilmerle/bt_models_cluster)
- cd bt_models_cluster
- pip install -r code/requirements.txt
- ./setup.sh
- ./run_models_python.sh
  - Falls das Python in einem virtualenv ist: ./run_models_virtualenv.sh
  - Gegebenenfalls muss der Pfad des virtualsenvs in run_models_virtualenv.sh angepasst werden

## Daten exportieren
- Am wichtigsten ist der Ordner output_data
- Falls ein Ordner "runs" erstellt wurde, gerne auch speichern