#!/usr/bin/env python
# coding: utf-8

# Predicting the quality of a wine by tasting is considered a very difficult task. Apparently, even professional wine tasters have an accuracy of only $71 /%$. In this problem, we will try to predict the quality of wine using knn regression and see if we can do better than professional wine tasters!
# 
# The dataset contains several parameters that describe the wine. The outcome variable is wine quality on a scale of $1$ to $10$. The goal is to predict the quality of wine using the avaiable features. 
# 
# - fixed acidity
# - volatile acidity
# - citric acid
# - residual sugar
# - chlorides
# - free sulfur dioxide
# - total sulfur dioxide
# - density
# - pH
# - sulphates
# - alcohol
# 
# Output variable (based on sensory data):
# - quality (score between 0 and 10)


# #### Step 0. Import the required libraries and functions

# In[ ]:


# numpy, pandas, matplotlib, seaborn, Kfold, cross_val_score, KneighborsRegressor, StandardScaler.


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np


# #### Step 1. Load the data and explore the dataset. You can also create a histograms of all the variables. Notice the histogram of wine quality. 

# In[5]:


# Step 1: Load the data
wine = pd.read_csv("winequality-red.csv")  

# Explore the dataset
print("\nFirst few rows of the dataset:\n", wine.head())
print("\nData type of the dataset:\n", wine.dtypes)
print("\nNullvalues of the dataset:\n", wine.isnull().sum())

# Summary statistics
print("\nSummary statistics:\n", wine.describe())

# Step 2: Create histograms of all variables
wine.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


# #### Step 2. Find the most correlated features with the wine quality

# In[6]:


# Calculate correlations
df_corr = wine.corr()
round(df_corr,2)

# Sort correlations in descending order
abs(df_corr[['quality']]).sort_values(by = 'quality', ascending = False).plot(kind='bar')

# Print the plot
plt.show()


# In[ ]:


#### The features that are most correlated with wine quality are:
alcohol, volatile acidity, sulphates,citric acid,total sulfur dioxide, free sulfur dioxide, density


# #### Step 3. Create X and y vectors. Scale the features by using the standard scalar. To do so, you need to create a scaler object and use the function *fit_transform*, and apply it on the array of features. Here's an example on how to do this:

# In[25]:


# Features that are most correlated with wine quality
selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 
                     'total sulfur dioxide', 'free sulfur dioxide', 'density']

# Create X and y vectors
X = wine[selected_features].values
y = wine['quality'].values

# define min max scaler
scaler = StandardScaler()
# transform data
X_scaled = scaler.fit_transform(X)
print(X_scaled)


# In[ ]:





# #### Step 4. Choose the best $k$ by using cross validation. Make sure to use the same cross validation splits for each evaluation of $k$ using the KFold function. Also, use the scaled features that you created in Step 3

# In[26]:


# Initialize KFold with 5 splits and set random_state for reproducibility
cvdata = KFold(n_splits=5, random_state=0, shuffle=True)

# Initialize a list to store R-squared values for each k
Rsquared = []

# Loop through each value of k from 1 to 50
for k in range(1, 51):   
    knn = KNeighborsRegressor(n_neighbors=k)
    # Perform cross-validation with KFold object
    scores = cross_val_score(knn, X_scaled, y, cv=cvdata)
    # Append the mean of cross-validation scores to Rsquared list
    Rsquared.append(scores.mean())  

# Plot the R-squared values against k
plt.plot(range(1, 51), Rsquared)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean R-squared')
plt.title('Cross-Validation for KNN Regression')
plt.show()

# Print the maximum R-squared and corresponding k value
print("Maximum R-squared:", max(Rsquared))
print("Corresponding k value:", range(1, 51)[np.argmax(Rsquared)])

# In[ ]:


##The R-squared value for the best model (k=50) is approximately 33%
#the wine tasters can predict with 71% accuracy
#the current model cannot replace wine tasters


# ### Find the R^2 for a linear regression model using k fold cross validation. Is KNN better than Linear regression?

# In[27]:


from sklearn.linear_model import LinearRegression
linearReg_pipeline = make_pipeline(StandardScaler(), LinearRegression()) # This is where we create the LR model
scores = cross_val_score(linearReg_pipeline, X_scaled, y, cv=cvdata) # This is where we do the k fold cross validation
scores.mean()


# In[ ]:


#Linear regresseion can predict the variation with 33% accuracy. 
#KNN can predict it with 31% accuracy
#linear regression is better

