import threading
import time
import numpy as np
import os
import cv2
import mediapipe as mp
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import tkinter.messagebox as msg
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
data_Words = os.listdir(DATA_PATH)
actions = np.array(data_Words)
no_sequences = 20  # folder numbers
sequence_length = 20  # arrary numbers
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

model = load_model('actions.h5')
model.load_weights('actions_weights.h5')

global sentence
sentence = []
sequence = []
threshold = 0.5

pTime = 0
cTime = 0

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]
        x1 = int(0.6 * image.shape[1])
        y1 = 200
        x2 = image.shape[1] - 30
        y2 = int(0.7 * image.shape[1])
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]


        def traducir():
            global sentence
            try:
                if res[np.argmax(res)].any() > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
            except IndexError:
                msg.showerror('Error', 'Se produjo un error')
                pass


        def mostrar():
            global sentence
            if len(sentence) > 1:
                sentence = sentence[-1:]

            cv2.rectangle(image, (0, 0), (640, 40), (225, 140, 25), -1)
            if sentence == ['Fondo Vacio']:
                cv2.putText(image, text=' ', org=(3, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            else:
                cv2.putText(image, text=' '.join(sentence), org=(3, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                print(actions[np.argmax(res)])


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        tr = threading.Thread(target=traducir, args=())
        ms = threading.Thread(target=mostrar, args=())

        tr.start()
        time.sleep(0.001)
        ms.start()
        ms.join()

        cv2.putText(image, str(int(fps)), (3, 80), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 51, 36),
                    thickness=2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 145, 10), thickness=2)
        cv2.imshow('Image', image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
