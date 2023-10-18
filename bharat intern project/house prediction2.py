#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('D:\house predication\Housing.csv')

# Prepare the dataset as you did in your previous code
# For encoding categorical variables as integers
data['mainroad'] = data['mainroad'].astype('category')
data['mainroad'] = data['mainroad'].cat.codes
data['guestroom'] = data['guestroom'].astype('category')
data['guestroom'] = data['guestroom'].cat.codes
data['basement'] = data['basement'].astype('category')
data['basement'] = data['basement'].cat.codes
data['hotwaterheating'] = data['hotwaterheating'].astype('category')
data['hotwaterheating'] = data['hotwaterheating'].cat.codes
data['airconditioning'] = data['airconditioning'].astype('category')
data['airconditioning'] = data['airconditioning'].cat.codes
data['prefarea'] = data['prefarea'].astype('category')
data['prefarea'] = data['prefarea'].cat.codes

# Define features and target
features = ['area', 'bedrooms', 'bathrooms', 'mainroad', 'parking']
X = data[features]
X.columns = features
y = data['price']

# Train a Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Collect user data
user_area = float(input("Enter the area of the property (in square feet): "))
user_bedrooms = int(input("Enter the number of bedrooms: "))
user_bathrooms = int(input("Enter the number of bathrooms: "))
user_mainroad = input("Is the property near the main road? (yes/no): ").strip().lower()
user_parking = input("Is there a parking space available? (yes/no): ").strip().lower()

# Validate user inputs for 'user_mainroad' and 'user_parking'
if user_mainroad not in ["yes", "no"]:
    print("Invalid input for 'user_mainroad'. Please enter 'yes' or 'no'.")
    exit()

if user_parking not in ["yes", "no"]:
    print("Invalid input for 'user_parking'. Please enter 'yes' or 'no'.")
    exit()

# Encode user data
user_data = np.array([user_area, user_bedrooms, user_bathrooms, int (user_mainroad == 'yes'), int (user_parking == 'yes')], dtype = object)

# Make a prediction for the user data
user_prediction = model.predict([user_data])

# Calculate the Euclidean distance between the user data and all data points
distances = [np.linalg.norm(point - user_data) for point in X.to_numpy()]

# Find the index of the data point with the minimum distance
closest_point_idx = np.argmin(distances)

# Get the corresponding data point
closest_point = X.iloc[closest_point_idx]

print("Closest data point to user input:")
print(closest_point)

user_df = pd.DataFrame({
    'User Data': [user_area, user_bedrooms, user_bathrooms, user_mainroad, user_parking],
    'Closest Actual Data': closest_point
})

# Print the user data and closest actual data
print("User Data and Closest Actual Data:")
print(user_df)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y, model.predict(X), c='blue', marker='.', label='Actual Data', s = 10)
plt.scatter([user_prediction], [user_prediction], c='red', marker='s', label='User Data')
plt.scatter([y.iloc[closest_point_idx]], [model.predict(X)[closest_point_idx]], c='green', marker='x', label='Closest Data Point')

# Plot a line connecting the user data to the closest data point
plt.plot([user_prediction, y.iloc[closest_point_idx]], [user_prediction, model.predict(X)[closest_point_idx]], 'k--')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.legend()
plt.grid()
plt.show()


# 
