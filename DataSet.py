import cv2
import mediapipe as mp
import os
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))


def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([rh, lh])


DATA_PATH = os.path.join('DATA_SET')
# actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
#                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
#                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Hola',
#                     'Como Estas', 'Mi', 'Estoy Bien', 'Mal', 'Lo Siento',
#                     'Te Amo', 'Gracias', 'Adios', 'Nombre es', 'Si', 'No', 'Fondo Vacio'])
actions = np.array(['U'])
no_sequences = 30  # folder numbers
sequence_length = 30  # arrary numbers

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holitic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        image, results = mediapipe_detection(frame, holitic)
        draw_landmarks(image, results)
        cv2.putText(image, 'Welcome User', (200, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(image, 'Press ESC to Start Capturing', (70, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1,
                    cv2.LINE_AA)
        x1 = int(0.6 * image.shape[1])
        y1 = 200
        x2 = image.shape[1] - 30
        y2 = int(0.7 * image.shape[1])
        roi = image[y1:y2, x1:x2]
        # cv2.imshow('Camara', image)
        image2 = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 145, 10), thickness=2)
        cv2.imshow('imagen', image2)
        cv2.imshow('roi', roi)
        key = cv2.waitKey(1) & 0xFF
        if (key == 27) or (key == ord('q')):
            break

cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                x1 = int(0.6 * image.shape[1])
                y1 = 200
                x2 = image.shape[1] - 30
                y2 = int(0.7 * image.shape[1])
                roi = image[y1:y2, x1:x2]
                if frame_num == 0:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 145, 10), thickness=2)
                    cv2.putText(image, 'Starting to Capture...', (150, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(image, 'Frames for {} Number {}'.format(action, sequence), (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.imshow('Image Collecting', image)
                    cv2.waitKey(1000)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 145, 10), thickness=2)
                    cv2.putText(image, 'Capturing... {}'.format(frame_num), (15, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(image, 'Frames for {} Number {}'.format(action, sequence), (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.imshow('Image Collecting', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()
