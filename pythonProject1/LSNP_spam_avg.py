import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import time
import Model
import Generator
import Classifier
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
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

    maxgen = 100

    classifier = Classifier.Classifier(train_x, train_y)
    model = Model.Model()

    model_w, model_l = classifier.train_classification(train_x, train_y)
    acc = []
    it = []
    for i in range(maxgen):
        it.append(i+1)
        model_w, model_l = classifier.train_classification(train_x, train_y)
        predict = model.classification(model_w, model_l, train_x, train_y)
        acc_train = model.accuracy_rate(predict, train_y)
        acc.append(acc_train)
        if acc_train > 0.93:
            break


    predict = model.classification(model_w, model_l, test_x, test_y)
    accuracy = accuracy_score(test_y, predict)
    precision = precision_score(test_y, predict, average='weighted')
    recall = recall_score(test_y, predict, average='weighted')
    f1 = f1_score(test_y, predict, average='weighted')
    return it[-1], accuracy, precision, recall, f1


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



