import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

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

# Assuming X is your feature matrix and y is the one-hot encoded labels
# Modify the input_dim and output_dim based on the number of features and classes
input_dim = X.shape[1]
output_dim = y.shape[1]  # Number of classes (should be 2 for binary classification)

tf.random.set_seed(42)

# Build the neural network model
model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(train_x, train_y, epochs=50, validation_data=(test_x, test_y))

plt.plot(history.history['accuracy'])
#plt.ylim(0.99, 1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
y_pred_prob = model.predict(test_x)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

# Convert one-hot encoded labels to class labels
y_true = np.argmax(test_y, axis=1)

# Calculate and print metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')  # Use 'micro', 'macro', or 'weighted' for multi-class
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
