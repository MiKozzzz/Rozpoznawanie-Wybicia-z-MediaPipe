import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def moving_average(data, window_size):
    """
    Obliczanie średniej ruchomej
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def is_point_on_line(p, line_start, line_end):
    """
    Sprawdza, czy punkt p leży na linii wyznaczonej przez line_start i line_end.
    """
    px, py = p
    x1, y1 = line_start
    x2, y2 = line_end
    a = (y1-y2)/(x1-x2)
    b = y1 - x1*a
    funkcja = px*a + b
    roznica = abs(py - funkcja)
    print(roznica)
    # 20 pixeli w góre w dół od prawidłowej wartości dla danej funkcji
    if roznica < 20:
        return True
    else:
        return False


def calculate_angle(a, b, c):
    """
    Oblicza kąt między trzema punktami a, b, c w stopniach.
    a, b, c - współrzędne punktów w formacie [x, y, z]
    """
    # Wektory AB i BC
    ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
    # Iloczyn skalarny i długości wektorów
    dot_product = sum(ab[i] * bc[i] for i in range(3))
    norm_ab = math.sqrt(sum(ab[i] ** 2 for i in range(3)))
    norm_bc = math.sqrt(sum(bc[i] ** 2 for i in range(3)))
    # Unikaj dzielenia przez zero
    if norm_ab == 0 or norm_bc == 0:
        return 0
    # Kąt w radianach
    angle_rad = math.acos(dot_product / (norm_ab * norm_bc))
    # Konwersja na stopnie
    angle_deg = math.degrees(angle_rad)
    return angle_deg


# Model rozpoznania progu
model = YOLO(r'C:\Milek\Programowanie\runs\detect\train2\weights\best.pt')

# Wczytywanie MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

video_path = r"C:\Milek\Programowanie\Rozpoznawanie_wybicia_skoczka\wybicie\4_58,2.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Błąd ładowania wideo!")
    exit()

kat = []
indeks = []
frame_index = 0  # Licznik klatek
frame_wybicia = 0
kat_wybicia = 0
while cap.isOpened():
    ret, frame = cap.read()
    # if not ret or frame_wybicia != 0:
    if not ret:
        break
    # Konwersja obrazu do RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Analiza obrazu przy pomocy MediaPipe Pose
    results = pose.process(image_rgb)
    # Wykonaj predykcję na obrazie
    # result_prog = model(frame, verbose=False, imgsz=320) # Troszke szybszy
    result_prog = model(frame, verbose=False)
    if results.pose_landmarks:
        # Rysowanie punktów ciała na obrazie
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        # Pobranie współrzędnych biodra, kolana i kostki (prawa strona)
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        # Wyświetlanie współrzędnych
        # print(f"Biodro: ({hip.x:.2f}, {hip.y:.2f}, {hip.z:.2f})")
        # print(f"Kolano: ({knee.x:.2f}, {knee.y:.2f}, {knee.z:.2f})")
        # print(f"Kostka: ({ankle.x:.2f}, {ankle.y:.2f}, {ankle.z:.2f})")
        hip = [hip.x, hip.y, hip.z]
        knee = [knee.x, knee.y, knee.z]
        ankle = [ankle.x, ankle.y, ankle.z]
        angle = calculate_angle(hip, knee, ankle)
        # print(f"Kąt w kolanie: {angle:.2f}°")
        kat.append(angle)
        indeks.append(frame_index)
        for result in result_prog[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]  # Współrzędne prostokąta
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Rysowanie prostokąta
            if frame_wybicia == 0:
                # Pobranie współrzędnych stóp
                right_foot = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
                left_foot = landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
                # Współrzędne stopy są znormalizowane, dlatego trzeb je pomnożyć z rozmiarami obrazu.
                right_foot_coords = (right_foot.x * frame.shape[1], right_foot.y * frame.shape[0])
                left_foot_coords = (left_foot.x * frame.shape[1], left_foot.y * frame.shape[0])
                # Sprawdzenie przecięcia z przekątną
                diagonal_start = (x1, y2)
                diagonal_end = (x2, y1)
                if (is_point_on_line(right_foot_coords, diagonal_start, diagonal_end) or
                        is_point_on_line(left_foot_coords, diagonal_start, diagonal_end)):
                    print(f"Przecięcie! Frame index: {frame_index}")
                    frame_wybicia = frame_index
                    kat_wybicia = angle

    # Zmniejsz rozmiar klatki (np. 640x480)
    frame_resized = cv2.resize(frame, (1280, 960))
    # Wyświetlanie obrazu
    cv2.imshow("Ski Jump Pose Analysis", frame_resized)
    frame_index += 1
    # Naciśnij 'q', aby zamknąć okno
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Zwalnianie zasobów
cap.release()
cv2.destroyAllWindows()

if frame_wybicia != 0:
    cap = cv2.VideoCapture(video_path)
    # Sprawdź, czy plik wideo został poprawnie otwarty
    if not cap.isOpened():
        print("Błąd przy otwieraniu pliku wideo!")
        exit()
    # Ustawienie numeru klatki
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_wybicia)
    # Odczytanie wybranej klatki
    ret, frame = cap.read()
    if ret:
        # Konwersja obrazu do RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Analiza obrazu przy pomocy MediaPipe Pose
        results = pose.process(image_rgb)
        # Wykonaj predykcję na obrazie
        result_prog = model(frame, verbose=False)
        # Rysowanie punktów ciała na obrazie
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for result in result_prog[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]  # Współrzędne prostokąta
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Rysowanie prostokąta
        # Wyświetlenie klatki
        print(f"FRAME: {frame_wybicia} KĄT WYBICIA: {kat_wybicia}")
        # Zmniejsz rozmiar klatki (np. 640x480)
        frame_resized = cv2.resize(frame, (1280, 960))
        cv2.imshow(f"FRAME: {frame_wybicia} KĄT WYBICIA: {kat_wybicia}", frame_resized)
        cv2.waitKey(0)  # Poczekaj na naciśnięcie klawisza
    else:
        print(f"Nie udało się odczytać klatki {frame_wybicia}")
    # Zwolnienie zasobów
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Nie udało się znaleźć momentu wybicia")

# Wygładzenie kąta za pomocą średniej ruchomej
window_size = 10  # Liczba sąsiednich punktów do uśredniania
smoothed_angles = moving_average(kat, window_size)

plt.figure(figsize=(10, 5))
plt.plot(indeks, kat, label="Kąt w kolanie", color="blue")
plt.plot(indeks[0:99], smoothed_angles, label="Kąt w kolanie wygładzony", color="green")
plt.scatter(frame_wybicia, kat_wybicia, label="Moment wybicia", color="red")
plt.xlabel("Klatka")
plt.ylabel("Kąt (stopnie)")
plt.title("Zmiana kąta w kolanie w czasie")
plt.legend()
plt.grid()
plt.show()


