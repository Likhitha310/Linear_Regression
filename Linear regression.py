#!/usr/bin/env python
# coding: utf-8

# # Importing all the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Assign the dataset to the DataFrame

# In[2]:


var_x = np.array([1.1,1.3,1.5,2.0,2.2,2.9,3.0, 3.2, 3.2, 3.7,3.9,4.0, 4.0, 4.1,4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3,10.5, 6.8, 7])
var_y = np.array([39343, 46205,37731, 43535, 39821, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 73504, 79123, 83088, 81363, 93940, 91738, 98217, 101302, 113812, 109431, 105582, 116969, 12635, 122391, 121872])


# In[3]:


X=  var_x.reshape(-1, 1)
y = var_y


# In[4]:


len(var_y)


# In[5]:


plt.scatter(var_x,var_y)


# In[6]:


df = pd.DataFrame({"Experience":var_x,"Salary":var_y})


# In[7]:


df.head()


# In[8]:


df.info()


# # Form a variable 'X' to store all the features and 'y' to store the target

# In[9]:


X = df.Experience
y = df.Salary


# # Splitting the data into two sets - Training Set & Testing Set

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# # Choose the model

# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


model = LinearRegression()


# # Train the model on the training set

# In[14]:


model.fit(X_train, y_train)


# # Make Predictions

# In[15]:


y_test


# In[16]:


y_pred = model.predict(X_test)


# # Performance Metric - MSE

# In[17]:


from sklearn.metrics import mean_squared_error


# In[18]:


mean_squared_error(y_test,y_pred)


# # Accuracy

# In[19]:


model.score(X_test, y_test)


# In[20]:


from sklearn.metrics import r2_score


# In[21]:


r2_score(y_test,y_pred)


# In[22]:


model.predict([[1]])

