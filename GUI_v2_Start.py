# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_v2_Start.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
################################################################
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from GUI_v2_Options import Ui_OptionWindow
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QPushButton
from PyQt5.QtWidgets import QStyleFactory
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import threading
import time
import numpy as np
import os
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
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))


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
global cap
################################################################


# noinspection PyAttributeOutsideInit,PyArgumentList,PyMethodMayBeStatic
class Ui_SignalTranslator(object):
    def setupUi(self, SignalTranslator):
        SignalTranslator.setObjectName("SignalTranslator")
        SignalTranslator.resize(686, 526)
        SignalTranslator.setStyleSheet("background-color:rgb(0, 170, 255)")
        self.widget = QtWidgets.QWidget(SignalTranslator)
        self.widget.setGeometry(QtCore.QRect(70, 60, 531, 401))
        self.widget.setStyleSheet("background-color: rgb(0, 255, 255);")
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(10, 10, 510, 380))
        self.label.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(SignalTranslator)
        self.label_2.setGeometry(QtCore.QRect(160, 20, 321, 31))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(SignalTranslator)
        self.pushButton.setGeometry(QtCore.QRect(600, 470, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color:rgb(0, 255, 0)")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(SignalTranslator)
        self.pushButton_2.setGeometry(QtCore.QRect(520, 470, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color:rgb(0, 255, 0)")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(SignalTranslator)
        self.pushButton_3.setGeometry(QtCore.QRect(430, 470, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color:rgb(0, 255, 0)")
        self.pushButton_3.setObjectName("pushButton_3")

        self.retranslateUi(SignalTranslator)
        QtCore.QMetaObject.connectSlotsByName(SignalTranslator)

        ################################################################
        self.pushButton.clicked.connect(SignalTranslator.close)
        self.pushButton_2.clicked.connect(self.cancel)
        self.pushButton_3.clicked.connect(self.start_video)

    def start_video(self):
        self.Work = Work()
        self.Work.start()
        self.Work.Imageupd.connect(self.Imageupd_slot)

    def Imageupd_slot(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    def cancel(self):
        self.label.clear()
        self.Work.stop()

    ################################################################
    def retranslateUi(self, SignalTranslator):
        _translate = QtCore.QCoreApplication.translate
        SignalTranslator.setWindowTitle(_translate("SignalTranslator", "Signal Translator App"))
        self.label_2.setText(_translate("SignalTranslator", "SIGNAL TRANSLATOR"))
        self.pushButton.setText(_translate("SignalTranslator", "Back"))
        self.pushButton_2.setText(_translate("SignalTranslator", "Stop"))
        self.pushButton_3.setText(_translate("SignalTranslator", "Start"))


########################################################
# noinspection PyAttributeOutsideInit
class Work(QThread):
    Imageupd = pyqtSignal(QImage)

    def run(self):
        global cap
        sequence = []
        cont = 0
        self.hilo_corriendo = True
        cap = cv2.VideoCapture(1)
        while self.hilo_corriendo:
            with mp_holistic.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as holistic:
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
                    cont = cont + 1
                    cv2.rectangle(image, (0, 0), (640, 40), (167, 122, 255), -1)  # visualizar palabras
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(55, 125, 34), thickness=2)  # encuadre de mano
                    if cont > 2:
                        words = actions[np.argmax(res)]
                        if words == 'Fondo Vacio':
                            cv2.putText(image, text='', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                            cont = 0
                        else:
                            cv2.putText(image, text=''.join(words), org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        print(words)
                    else:
                        cont = cont + 1

                convertir_QT = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pic = convertir_QT.scaled(510, 540, Qt.KeepAspectRatio)
                self.Imageupd.emit(pic)

    def stop(self):
        global cap
        self.hilo_corriendo = False
        cap = cv2.VideoCapture(1)
        while self.hilo_corriendo:
            cap.release()
            self.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('windowsvista')  # ['windowsvista', 'Windows', 'Fusion']
    w = QMainWindow()
    ui = Ui_SignalTranslator()
    ui.setupUi(w)
    w.setFixedWidth(686)
    w.setFixedHeight(526)
    w.show()
    sys.exit(app.exec_())