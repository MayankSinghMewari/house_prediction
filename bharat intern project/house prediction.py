#!/usr/bin/env python
# coding: utf-8

# In[158]:


# Import necessary libraries


# In[159]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


# In[160]:


# we will Load the dataset here with this set of code


# In[161]:


data = pd.read_csv('D:\house predication\Housing.csv')
data


# In[162]:


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

data['furnishingstatus'] = data['furnishingstatus'].astype('category')
data['furnishingstatus'] = data['furnishingstatus'].cat.codes

data


# In[163]:


# Define features and target by which we have to pridict the best result


# In[164]:


X = data[['area', 'bedrooms', 'bathrooms', 'mainroad', 'parking']]
y = data['price']


# In[165]:


# we will Split the data into training and testing sets


# In[166]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[167]:


# Create a Linear Regression model


# In[168]:


model = LinearRegression()


# In[169]:


# Train the model


# In[170]:


model.fit(X_train, y_train)


# In[171]:


# Make predictions


# In[172]:


y_pred = model.predict(X_test)


# In[173]:


# Evaluate the model


# In[174]:


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[175]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Prediction Line')
plt.title(f"Actual Prices vs. Predicted Prices (MSE: {mse:.2f})")
plt.grid()
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




