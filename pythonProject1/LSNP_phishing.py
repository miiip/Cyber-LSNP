import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import time
import Model
import Generator
import Classifier
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('phishing_legitimate_full.csv')

y = df['CLASS_LABEL']
X = df.drop('CLASS_LABEL', axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
encoder = OneHotEncoder()
y = np.array(y).reshape(-1, 1)
y = encoder.fit_transform(y).toarray()
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

maxgen = 50

classifier = Classifier.Classifier(train_x, train_y, 42)
model = Model.Model()

model_w, model_l = classifier.train_classification(train_x, train_y)
predict = model.classification(model_w, model_l, train_x, train_y)
acc_train_last = model.accuracy_rate(predict, train_y)
acc = []
it = []
for i in range(maxgen):
    it.append(i+1)
    model_w, model_l = classifier.train_classification(train_x, train_y)
    predict = model.classification(model_w, model_l, train_x, train_y)
    acc_train = model.accuracy_rate(predict, train_y)
    acc.append(acc_train)
plt.plot(it, acc)
#plt.ylim(0.99, 1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
predict = model.classification(model_w, model_l, test_x, test_y)
print('Samples', len(test_x))
accuracy = accuracy_score(test_y, predict)
precision = precision_score(test_y, predict, average='weighted')
recall = recall_score(test_y, predict, average='weighted')
f1 = f1_score(test_y, predict, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")



