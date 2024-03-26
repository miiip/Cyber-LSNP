import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


def experiment(X, y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    model = Sequential()
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()

    early_stopping_callback = EarlyStopping(monitor='accuracy', min_delta=0, patience=0, mode='max', baseline=0.99)

    history = model.fit(train_x, train_y, epochs=100, verbose=0, validation_data=(test_x, test_y),
                        callbacks=[early_stopping_callback])
    end_time = time.time()
    elapsed_time = end_time - start_time

    y_pred_prob = model.predict(test_x)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

    y_true = np.argmax(test_y, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')  # Use 'micro', 'macro', or 'weighted' for multi-class
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return len(history.epoch), accuracy, precision, recall, f1


df = pd.read_csv('phishing_legitimate_full.csv')

y = df['CLASS_LABEL']
X = df.drop('CLASS_LABEL', axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
encoder = OneHotEncoder()
y = np.array(y).reshape(-1, 1)
y = encoder.fit_transform(y).toarray()

num_epchs = []
accuracies = []
precisions = []
recalls = []
f1s = []
for trial in range(20):
    num_epoch, acc, p, r, f1 = experiment(X, y)
    num_epchs.append(num_epoch)
    accuracies.append(acc)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

print(f"Num epochs: {np.mean(np.array(num_epchs)):.4f}")
print(f"Accuracy: {np.mean(np.array(accuracies)):.4f}")
print(f"Precision: {np.mean(np.array(precisions)):.4f}")
print(f"Recall: {np.mean(np.array(recalls)):.4f}")
print(f"F1 Score: {np.mean(np.array(f1s)):.4f}")




