# 🎿 Rozpoznawanie Wybicia z MediaPipe

Projekt wykorzystujący **YOLOv8** oraz **MediaPipe Pose** do analizy momentu wybicia w skokach narciarskich na podstawie nagrań wideo z boku.

## 🔍 Opis

Celem projektu jest wykrycie momentu wybicia skoczka narciarskiego na podstawie analizy wideo:

- **YOLOv8** służy do detekcji **progu** na nagraniu.
- **MediaPipe** analizuje **sylwetkę skoczka**, skupiając się na:
  - Położeniu kostki, kolana i biodra,
  - Wyznaczeniu kąta w stawie kolanowym.
- Program identyfikuje moment przecięcia stopy skoczka z przekątną progu i oznacza ten moment jako **wybicie**.


Dodatkowo generowany jest wykres kąta kolana w czasie z zaznaczonym momentem wybicia.

## 🧠 Technologie

- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose)
- OpenCV
- Matplotlib
- NumPy
- Python 3.10+

## 🗂️ Struktura projektu
- wybicia/ - Nagrania momentu rozbiegu skoczków
- train/ - Zbiór treningowy dla YOLO
- test/ - Zbiór testowy dla YOLO
- valid/ - Zbiór walidacyjny dla YOLO
- best.pt - Wytrenowany model YOLO wykrywający próg
- data.yaml - Konfiguracja danych do trenowania YOLO
- Rozpoznawanie_wybicia.py - Główny skrypt analizy wideo
- README.md - Opis projektu

## ⚙️ Wymagania

- Python 3.8+
- OpenCV
- MediaPipe
- matplotlib
- numpy
- [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8)

## 📊 Przykład wykresu

Program generuje wykres kąta zgięcia kolana z zaznaczonym momentem wybicia (czerwona kropka):

<img width="500" height="250" alt="Figure_1" src="https://github.com/user-attachments/assets/edb95ff1-5757-46bf-a111-b348498d6c2a" />


## 📸 Przykład klatki momentu wybicia

<img width="450" height="450" alt="obraz" src="https://github.com/user-attachments/assets/eec45799-a1ec-4afe-8851-591e508f2ab3" />


## 📄 Licencja

Projekt edukacyjny.

