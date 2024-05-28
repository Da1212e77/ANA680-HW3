import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt

# Load data
red_wine = pd.read_csv('winequality-red.csv')
white_wine = pd.read_csv('winequality-white.csv')

# Add type column
red_wine['type_red'] = 1
red_wine['type_white'] = 0
white_wine['type_red'] = 0
white_wine['type_white'] = 1

# Combine data
wine = pd.concat([red_wine, white_wine])

# Split data into features and target
X = wine.drop(columns=['quality'])
y = wine['quality']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(y, model.predict(X))
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()

residuals = y - model.predict(X)
plt.figure(figsize=(12, 8))
plt.scatter(model.predict(X), residuals)
plt.xlabel('Predicted Quality')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.savefig('residual_plot.png')
plt.show()
