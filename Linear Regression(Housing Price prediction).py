#!/usr/bin/env python
# coding: utf-8

# In[34]:


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


# In[35]:


warnings.filterwarnings("ignore", category=UserWarning)


# # 1.Import Libraries

# Standrad Libraries

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# Scikit-learn modules

# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing


# # Load the California housing dataset from sklearn

# In[5]:


# Load the California Housing Prices dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)


# # Exploratory Data Analysis

# In[13]:


df.shape


# In[7]:


df['MedHouseVal'] = data.target


# In[8]:


df.head()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[14]:


df.isnull().sum()


# In[21]:


df.columns


# In[23]:


df.dtypes


# In[25]:


df.nunique().sum()


# In[24]:


# Display unique values for each column in the DataFrame
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in '{column}': {unique_values[:10]}")  # Displaying first 10 unique values for brevity
    print(f"Number of unique values in '{column}': {len(unique_values)}\n")


# In[43]:


#minimum price
df.MedHouseVal.min()


# In[44]:


#maximum price
df.MedHouseVal.max()


# In[45]:


#standrad deviation
df.MedHouseVal.std()


# In[16]:


# Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)


# In[28]:


# Define a custom color palette with flourish
flourish_palette = ['#FF85A2', '#FF9F84', '#FFAB73', '#FFC971', '#FFE17D']


# In[27]:


# Heatmap of the correlation matrix with the custom color palette
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap=flourish_palette, linewidths=0.5)
plt.title('Correlation Matrix Heatmap (Flourish Palette)')
plt.show()


# In[29]:


# Plot histograms for all features
# Histograms with custom color palette
df.hist(bins=50, figsize=(20,15), color=flourish_palette[0])
plt.tight_layout()
plt.show()


# In[30]:


# Scatter plot of features against the target variable with custom color palette
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    plt.scatter(df[feature], df['MedHouseVal'], alpha=0.5, color=flourish_palette[i % len(flourish_palette)])
    plt.title(f"{feature} vs MedHouseVal")
    plt.xlabel(feature)
    plt.ylabel('MedHouseVal')

plt.tight_layout()
plt.show()


# In[33]:


# Pairplot for selected features and target with custom color palette
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']], plot_kws={'color': flourish_palette[0]})
plt.show()


# In[32]:


# Distribution of the target variable with custom color palette
plt.figure(figsize=(10, 6))
sns.histplot(df['MedHouseVal'], bins=50, kde=True, color=flourish_palette[0])
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value ($100,000)')
plt.ylabel('Frequency')
plt.show()


# In[40]:


# Create a box plot to visualize outliers
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, orient='h')
plt.title('Box Plot of Housing Prices (Including Outliers)')
plt.xlabel('Price ($100,000)')
plt.show()


# In[111]:


# Density plot of the target variable
plt.figure(figsize=(8, 6))
sns.kdeplot(df['MedHouseVal'], shade=True, color='skyblue')
plt.title('Density Plot of Median House Value')
plt.xlabel('Median House Value ($100,000)')
plt.ylabel('Density')
plt.show()


# In[112]:


# Violin plot of median house value
plt.figure(figsize=(10, 6))
sns.violinplot(x=df['MedHouseVal'], color='salmon')
plt.title('Violin Plot of Median House Value')
plt.xlabel('Median House Value ($100,000)')
plt.show()


# In[113]:


# Bar plot of median income categories
plt.figure(figsize=(10, 6))
sns.countplot(x=pd.cut(df['MedInc'], bins=5), palette='pastel')
plt.title('Bar Plot of Median Income Categories')
plt.xlabel('Median Income Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[114]:


# Joint plot of median income and median house value
sns.jointplot(x='MedInc', y='MedHouseVal', data=df, kind='hex', color='lavender')
plt.title('Joint Plot of Median Income and Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value ($100,000)')
plt.show()


# In[ ]:





# # Export The Dataset

# In[36]:


# Export the DataFrame to a CSV file
df.to_csv('california_housing_dataset.csv', index=False)


# In[37]:


# Find the minimum and maximum prices
min_price = df['MedHouseVal'].min()
max_price = df['MedHouseVal'].max()

print(f"Minimum Price: ${min_price}")
print(f"Maximum Price: ${max_price}")


# In[38]:


# Find the difference between maximum and minimum prices
price_difference = max_price - min_price

print(f"Difference between Maximum and Minimum Prices: ${price_difference}")


# # MACHINE LEARNING-Linear Regression

# In[59]:


# Feature matrix X and target vector y
U = df.drop('MedHouseVal', axis=1)
V = df['MedHouseVal']


# In[60]:


U


# In[61]:


V


# In[77]:


# Feature matrix X and target vector y
X = np.array(df.drop('MedHouseVal', axis=1))
y = np.array(df['MedHouseVal'])


# In[78]:


X


# In[79]:


y


# In[80]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[81]:


len(X_train)


# In[82]:


len(y_train)


# In[83]:


len(X_test)


# In[84]:


len(y_test)


# Choose the model - Linear Regression

# In[85]:


# Initialize and train the linear regression model
model = LinearRegression()


# In[86]:


#Fitting/Train the model
model.fit(X_train, y_train)


# In[87]:


model.intercept_


# In[88]:


model.coef_


# In[89]:


# Predict housing prices on the testing set
y_pred = model.predict(X_test)


# In[90]:


y_pred


# # Testing the model performance

# In[91]:


#model score
model.score(X_test,y_test)


# In[94]:


#R Squared
r2_score(y_test,y_pred)


# In[96]:


#mean Squared error
mean_squared_error(y_test, y_pred)


# In[97]:


#mean absolute error
mean_absolute_error(y_test, y_pred)


# In[101]:


#Root Mean Squared Error
np.sqrt(mean_squared_error(y_test, y_pred))


# In[102]:


#Adjusted R Squared
# Calculate the number of samples (n) and number of predictors (p)
n = X.shape[0]  # Number of samples
p = X.shape[1]  # Number of predictors (features)

# Calculate the R-squared (R2) score of the model on the testing data
r2 = model.score(X_test, y_test)

# Calculate the adjusted R-squared
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"Adjusted R-squared (R2): {adjusted_r2}")


# In[110]:


# Scatter plot of actual versus predicted values 
plt.scatter(y_test, y_pred, color='grey')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid()
plt.plot(min(y_test),max(y_test),min(y_test),max(y_test), color = 'black')
plt.title('Actual vs Predicted Values')
plt.show()


# In[115]:


# Calculate residuals
residuals = y_test - y_pred


# In[116]:


# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='lavender')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[117]:


# Distribution of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='skyblue')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[118]:


# Q-Q plot
import statsmodels.api as sm

plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line='45', color='salmon')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()



# In[119]:


# Predict prices on the entire dataset (optional)
prices_pred = model.predict(X)
df['PredictedMedHouseVal'] = prices_pred


# In[120]:


df.head()


# # Interactive Web Dashboard

# In[135]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the California Housing Prices dataset
df = pd.read_csv('California Housing Prices.csv')

# Define the app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("California Housing Prices Prediction"),
    html.Label("Enter features:"),
    dcc.Input(id='med_inc', type='number', placeholder='Median Income'),
    dcc.Input(id='house_age', type='number', placeholder='House Age'),
    dcc.Input(id='ave_rooms', type='number', placeholder='Average Rooms'),
    dcc.Input(id='ave_bedrms', type='number', placeholder='Average Bedrooms'),
    dcc.Input(id='population', type='number', placeholder='Population'),
    dcc.Input(id='ave_occup', type='number', placeholder='Average Occupancy'),
    dcc.Input(id='latitude', type='number', placeholder='Latitude'),
    dcc.Input(id='longitude', type='number', placeholder='Longitude'),
    html.Br(),
    html.Button('Predict', id='predict-btn', n_clicks=0),
    html.Br(),
    html.Div(id='prediction-output')
])

# Define callback to update output
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [Input('med_inc', 'value'),
     Input('house_age', 'value'),
     Input('ave_rooms', 'value'),
     Input('ave_bedrms', 'value'),
     Input('population', 'value'),
     Input('ave_occup', 'value'),
     Input('latitude', 'value'),
     Input('longitude', 'value')]
)
def update_output(n_clicks, med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    if n_clicks > 0:
        # Prepare features for prediction
        features = [[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]]
        
        # Load the trained linear regression model
        model = LinearRegression()
        X = df.drop(columns=['MedHouseVal'])
        y = df['MedHouseVal']
        model.fit(X, y)
        
        # Make prediction
        prediction = model.predict(features)[0]
        return f"Predicted Housing Price: ${prediction:,.2f}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




