from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_PATH = os.path.join('DATA_SET')
data_Words = os.listdir(DATA_PATH)
actions = np.array(data_Words)
no_sequences = 30  # folder numbers
sequence_length = 30  # arrary numbers
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

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# print(X_train.shape)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
model.fit(X_train, y_train, epochs=180, callbacks=[tb_callback])
res = model.predict(X_test)
print(actions[np.argmax(res[1])])
print(actions[np.argmax(y_test[1])])
model.save('actions.h5')
model.save_weights('actions_weights.h5')
print('Modelo y pesos guardados exitosamente')
yhat = model.predict(X_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
Confusion_Matrix = multilabel_confusion_matrix(ytrue, yhat)
Accuracy = accuracy_score(ytrue, yhat)
