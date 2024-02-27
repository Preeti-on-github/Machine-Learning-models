#!/usr/bin/env python
# coding: utf-8

# # Problem 1: Predicting the quality of wines
# 
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
# 
# 
# You can follow the same steps that we did in class for the California Housing dataset. Here are the steps:
# 

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

# Create a KNN pipeline with StandardScaler and KNeighborsRegressor with k=13
#knn_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=13))
# Perform cross-validation with the pipeline
#scores = cross_val_score(knn_pipeline, X, y, cv=cvdata)
#print("Mean R-squared with pipeline:", scores.mean())


# ### What is the $R^2$ for the best model? Do you think this machine learning model can replace professional wine tasters?

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


# # Problem 2: Feature selection using cross validation
# 
# In this problem, we will take a different approach towards feature selection and choosing the best value of $k$. We will be working with the california housing dataset. Recall that in class we used the correlation between median prices and features to pick the most predictive features. Using correlation, we selected the following features:
# 
# - MedInc
# - HouseAge
# - AveRooms
# - Latitude
# - Longitude
# 
# Now we will do a more refined feature selection by using the $R^2$ of knn models with different features. We will fit 3 different model families:
# 1. knn with varying k, and features = 'Latitude' and 'Longitude'
# 
# 2. knn with varying k, and features = 'Latitude', 'Longitude', 'MedInc'
# 
# 3. knn with varying k, and features = 'Latitude', 'Longitude', 'MedInc' and 'HouseAge'
#  
# I will guide you through the process step by step
# 

# In[ ]:


#### Step 0: import the following functions: 
# numpy, pandas,  matplotlib, KFold, cross_val_score, KNeigbhorRegressor, StandardScaler


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np


# In[ ]:


#### Step 1: Load the california_housing.txt file


# In[14]:


housing = pd.read_csv("california_housing.txt")
housing.head()


# In[ ]:


#### Step 2: create a crossvalidation object using the Kfold function. Use 5 splits.


# In[15]:


cvdata = KFold(n_splits=5, random_state=0,shuffle=True)


# In[ ]:


#### Step 3: Now we will use the features latitutde and longitude only to predict median house price.
#We will find the best 'k' by using cross validation with features "Latitutde" and "longitude".
#We have to make sure to scale the data by using the standard scaler (Use the function fit_transform() for scaling)
#Here are the steps

## Step 3.1 Create a array of features 'latitude' and 'longitude' and scale them using a standard scaler. 
# You can also transform the y using np.log() function


# In[29]:


# Create X and y vectors
X1 = housing[['Latitude','Longitude']].values
y1 = housing[['MedHouseVal']].values

# define min max scaler
scaler = StandardScaler()
# transform data
X_scaled1 = scaler.fit_transform(X1)
print(X_scaled1)


# In[ ]:


## Step 3.2 Run a cross validation for k ranging from 1 to 50. Compute the Rsquared for each k and store it in a list.
#Make sure to use the scaled features, and also use the cross validation object created in step 2
 


# In[30]:


# Initialize a list to store R-squared values for each k
Rsquared_list1 = []

# Run a cross-validation for k ranging from 1 to 50
for k in range(1, 51):   
    knn = KNeighborsRegressor(n_neighbors=k)
    # Perform cross-validation with the KFold object created in step 2
    scores = cross_val_score(knn, X_scaled1, y1, cv=cvdata, scoring='r2')
    # Append the mean of cross-validation scores to Rsquared_list
    Rsquared_list1.append(scores.mean())  


# In[ ]:


# Step 3.3 Plot the values of k vs Rsquared. Find the best RMSE and the corresponding value of k


# In[31]:


# Plot R-squared values against k
plt.plot(range(1, 51), Rsquared_list1)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean R-squared')
plt.title('Cross-Validation for KNN Regression')
plt.show()

# Find the best Rsquared and corresponding k value
best_rsquare1 = max(Rsquared_list1)
best_k1 = Rsquared_list1.index(best_rsquare1) + 1
print(f"Best Rsquared: {best_rsquare1} (for k={best_k1})")


# In[ ]:


# Step 4: Now repeat all the steps in Step 3,but this time  using the features 'Latitude", 'Longitude', 'MedInc'


# In[32]:


# Create X and y vectors
X2 = housing[['Latitude','Longitude', 'MedInc']].values
y1 = housing[['MedHouseVal']].values

# define min max scaler
scaler2 = StandardScaler()
# transform data
X_scaled2 = scaler.fit_transform(X2)
print(X_scaled2)

Rsquared_list2 = []

# Run a cross-validation for k ranging from 1 to 50
for k in range(1, 51):   
    knn = KNeighborsRegressor(n_neighbors=k)
    # Perform cross-validation with the KFold object created in step 2
    scores = cross_val_score(knn, X_scaled2, y1, cv=cvdata, scoring='r2')
    # Append the mean of cross-validation scores to Rsquared_list
    Rsquared_list2.append(scores.mean())  

# Plot R-squared values against k
plt.plot(range(1, 51), Rsquared_list2)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean R-squared')
plt.title('Cross-Validation for KNN Regression')
plt.show()

# Find the best Rsquared and corresponding k value
best_rsquare2 = max(Rsquared_list2)
best_k2 = Rsquared_list2.index(best_rsquare2) + 1
print(f"Best Rsquared: {best_rsquare2} (for k={best_k2})")


# In[ ]:


## Step 5  Repeat the previous steps using the features 'latitute', 'longitude', 'MedInc' and 'HouseAge' 
#to find the best $k$ and $R^2$


# In[33]:


# Create X and y vectors
X3 = housing[['Latitude','Longitude', 'MedInc', 'HouseAge']].values
y1 = housing[['MedHouseVal']].values

# define min max scaler
scaler3 = StandardScaler()
# transform data
X_scaled3 = scaler.fit_transform(X2)
print(X_scaled3)

Rsquared_list3 = []

# Run a cross-validation for k ranging from 1 to 50
for k in range(1, 51):   
    knn = KNeighborsRegressor(n_neighbors=k)
    # Perform cross-validation with the KFold object created in step 2
    scores = cross_val_score(knn, X_scaled3, y, cv=cvdata, scoring='r2')
    # Append the mean of cross-validation scores to Rsquared_list
    Rsquared_list3.append(scores.mean())  

# Plot R-squared values against k
plt.plot(range(1, 51), Rsquared_list3)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean R-squared')
plt.title('Cross-Validation for KNN Regression')
plt.show()

# Find the best Rsquared and corresponding k value
best_rsquare3 = max(Rsquared_list3)
best_k3 = Rsquared_list2.index(best_rsquare3) + 1
print(f"Best Rsquared: {best_rsquare3} (for k={best_k3})")


# In[ ]:


#now you can fill in the following table:

#Best R^2 for Latitude, Longitude 0.7918079829504591
#Best R^2 for latitude, longitude, MedInc 0.7668428062352588
#best R^2 for latitude, longitude, MedInc, HouseAge 0.7668428062352588

#Which model is the best?

#model with just latitude and longitude

# What is the lesson learnt from this? Are more features always better? Specially for models like knn?
#its always best to test out features and its not always good to have multiple featues as it might make the model more complicated and also might lead to lower level of accuracy in predictions


# # Problem 3: Nuances of knn
# 
# In this problem, we will try to understand some situations where knn does not work very well. There are two main issues:
# 
# - Categorical features and the choice of distance 
# - The curse of dimensionality - Too many features
#  

# So far, we have only used knn when we have numeric features. When we have numeric features, we can use the eucledian distance. 
# 
# But What if we have categorical features as well? The eucledian distance does not make sense in the case of categorical features (Recall the defintion of eucledian distance from the previous HW).
# 
# If we have only categorical features, we can use edit distance, or hamming distance. We will do this in several steps.
# 
# First we need to code the categorical features as dummy variables. Here's an example:
# 

# In[34]:


import pandas as pd
cat = pd.read_csv('cat.csv')
cat.head()


# We use the get_dummies function to create dummy variables from categorical data

# In[35]:


dummy = pd.get_dummies(cat)
dummy.head()


# In[36]:


#Now we can access the first person by using the .loc method
dummy.loc[0]


# In[37]:


dummy.loc[1]


# The hamming distance between two strings is the number of positions in which they differ.
# 
# For example, the hamming distance between 110 and 111 is 1. The hamming distance between 100 and 111 is 2.

# In[38]:


def hammingDist(str1, str2):
    i = 0
    count = 0
 
    while(i < len(str1)):
        if(str1[i] != str2[i]):
            count += 1
        i += 1
    return count


# In[39]:


hammingDist(dummy.loc[0],dummy.loc[1])


# In[40]:


hammingDist(cat.loc[0],cat.loc[1])


# In case we only have categorical features, we can use distances such as hamming distance, edit distance and so on. But if we have both numeric and categorical features (called mixed features), then its a bit tricky. Can you think of how to compute the distance in KNN when you have both continuos and categorical fetures?

# In[ ]:


#your answer here
#We can first calculate the hamming distance for categorical and then scale all the features together.


# ## The curse of dimensionality 
# 
# In Problem 2, we saw that more features in knn does not neccessary mean a better model. 
# 
# This phenomena is a limitation of knn. When we have lots of features, knn generally performs poorly. This is called the curse of dimensionality. 
# 
# Mathematically, the term `dimension` is used to describe the number of features. For example, if there are 3 features, we are in 3 dimensional space, if we have 100 features, we are in 100 dimensional space. (we can only visualize in 2 or 3 dimensions). As the number of dimensions increase, there are some weird things that happen.
# 
# One of the weird things that happens is that in high dimensions `all points are far from each other`. So algorithms like nearest neighbors fail because given a single point, we cannot find any points "close" to it.
# 
# This happens because there is more "space" in high dimensions. (There is a lot more to this issue, but I am trying to explain it in the simplest possible manner). The following figure illustrates the idea:

# In the figure, you can see, as we move from 1 dimension to 2,3 and 4 dimensions, there is more space and the points become farther and farther.
# ![image.png](attachment:image.png)
