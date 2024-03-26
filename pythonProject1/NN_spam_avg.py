import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_glove_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = list(map(float, values[1:]))
            word_vectors[word] = vector
    return word_vectors


glove_model = load_glove_vectors('glove.6B.300d.txt')
def get_avg_word2vec(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    word_vectors = [glove_model[word] for word in tokens if word in glove_model]
    if not word_vectors:
        return [0] * 300  # If no valid word vectors are found, return a vector of zeros

    # Calculate the average vector
    avg_vector = np.mean(word_vectors, axis=0)
    return avg_vector



def experiment(X, y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    model = Sequential()
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()

    early_stopping_callback = EarlyStopping(monitor='accuracy', min_delta=0, patience=0, mode='max', baseline=0.93)

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


df = pd.read_csv('spam_ham_dataset.csv')
df = df.drop('label', axis=1)
df = df.rename(columns={'label_num': 'label'})
count_spam = (df['label'] == 1).sum()
count_ham = (df['label'] == 0).sum()
ham_records = df[df['label'] == 0]
rows_to_drop = ham_records.sample(n=count_ham - count_spam, random_state=42)  # Set a random_state for reproducibility
df = df.drop(rows_to_drop.index)
print((df['label'] == 1).sum())
print((df['label'] == 0).sum())

print(df['text'])
df['text'] = df['text'].apply(get_avg_word2vec)
print(df['text'])

X = df['text'].values.tolist()
y = df['label'].values.tolist()


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




