#!/usr/bin/env python
# coding: utf-8

# # Phong Hoang - Predicting Housing Price

# ## Importing Libraries

# In[59]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


# ## Importing the Data Set and Observe the Data Structure

# In[18]:


housing = pd.read_csv('housing.csv')
housing.head()


# In[20]:


housing.info()


# Notice that every coloumn has 20640 rows, excepts for the `total_bedrooms` column, which contains only 20433 rows. Therefore, this column has some missing values and we need a proper cleaning for it.

# ## Visualizing the data

# In[31]:


housing.hist(bins=50, figsize=(15,15))
plt.show()


# Focusing on the histogram of `total_bedrooms`, we see that it is a slightly right-skewed distribution, with the median can be interpreted as the represented value for the column; thus, a great way to deal with the NA value in this situation is using the median of it.

# ## Taking care of missing data

# In[34]:


median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True) 


# In[35]:


housing.info()


# We check again the number of non-null values in the `total_bedrooms` to see our replacement code correct. It appears that the column now is containing 20640 values, which means our missing values have been filled.

# ## Seperating the independent values and dependent values

# In[43]:


X = housing.iloc[:, housing.columns != 'median_house_value' ].values
y = housing.iloc[:, -5].values


# ## Looking for the correlation

# By simply use the `corr` method, we can compute the the standard correlation coefficient of each column with the `median_house_value`.

# In[44]:


corr = housing.corr()
corr["median_house_value"].sort_values(ascending=False)


# From the correlation of these values, it seems that only the `median_income` affects directly to the housing price as it holding a correlation of 0.688, while every column else has a value near to 0, meaning that there is no
# linear correlation with the `median_housing_income`. Therefore, we want to add some attribute to find the insights of the data.

# In[45]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
housing.head()


# In[46]:


corr = housing.corr()
corr["median_house_value"].sort_values(ascending=False)


# These new columns we add create a much better correlation with the median house values. Clearly, the less ratio of bedrooms per room, the more expensive the house will be or the room per house is an important factor in deciding the price of the house

# # Encoding the data

# When we check the information of the data again, the ocean_proximity has the data type of non-number, creating problems when we try to train a model for our data set. So, using `OneHotEncoder`, we can transform these category values to the vectors with numeric value.

# In[47]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[49]:


print(X)


# ## Splitting the dataset into the Training set and Test set

# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# ## Feature Scaling

# In[62]:


housing["total_rooms"].max()


# In[63]:


housing["total_rooms"].min()


# In[64]:


housing["median_income"].max()


# In[65]:


housing["median_income"].min()


# Our columns contains a different range of values, for example the columns `total_rooms`has a range from 2 to 39320, while the median income has a range between 0 and 15. Such different ranges can create problem for our Machine Learning algorithm to learn and train the data, so we need to apply a feature scaling called standardization into our data set.

# In[66]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 5:] = sc.fit_transform(X_train[:, 5:])
X_test[:, 5:] = sc.transform(X_test[:, 5:])


# # Select and Train a Model

# In[76]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
grid_search.fit(X_train, y_train)


# In[90]:


print(grid_search.best_estimator_)


# In[98]:


forest_reg = RandomForestRegressor(max_features = 8, n_estimators = 30)
forest_reg.fit(X_train, y_train)


# In[99]:


np.set_printoptions(threshold= np.inf)
y_pred = forest_reg.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# After choosing the RandomForestRegressor with the help of GridSearch in determining the best parameters, we will measure the model's RMSE to see our result.

# In[100]:


from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# We get a very promising RMSE of 49248. However, further research need to include to determine the best estimators for this data set and get a better RMSE socre. 
