import pandas as pd # type: ignore
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Take user input
print("\nEnter Iris Flower Measurements:")

sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))

# Create sample
sample = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predict
prediction = model.predict(sample)

# Display result
print("\nPredicted Flower Species:",
      iris.target_names[prediction[0]].capitalize())

# Display model accuracy
print(f"Model Accuracy: {accuracy * 100:.2f}%")
