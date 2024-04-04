import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Load the data
data = pd.read_csv("Final_Dataset.csv")

# Split data into features and target
X = data.drop("prognosis", axis=1)
y = data['prognosis']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = keras.Sequential([
    keras.layers.Input(shape=X_train_scaled.shape[1:]),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(len(set(y_train)), activation="softmax")
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=100, validation_split=0.2, verbose=1)

model.save("ANN_model.h5")

# Evaluate the model
score = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])