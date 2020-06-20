# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:58:36 2020

@author: ankur
"""
#Step 0 -- import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Step 1 -- Read the data
bikes = pd.read_csv('hours.csv')


# Step 2 -- Prelim Analysis and Feature Selection
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'],axis=1)

# Check nulls or missing values
bikes_prep.isnull().sum()

# Simple Visualization of the data using Pandas Histogram
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

# Notes
# Demand is not normally distributed.

#Data visualisation - continuous features vs demand
plt.subplot(2,2,1)
plt.title('Temperature vs Demand')
plt.scatter(bikes_prep['temp'],bikes['demand'],s=2, c='g')

plt.subplot(2,2,2)
plt.title('aTemp vs Demand')
plt.scatter(bikes_prep['atemp'],bikes['demand'],s=2, c='b')

plt.subplot(2,2,3)
plt.title('Humidity vs Demand')
plt.scatter(bikes_prep['humidity'],bikes['demand'],s=2, c='m')

plt.subplot(2,2,4)
plt.title('windspeed vs Demand')
plt.scatter(bikes_prep['windspeed'],bikes['demand'],s=2, c='c')
plt.tight_layout()

# Notes - Updated
# Demand is not normally distributed.
#Temperature and demand appears to have direct correlation.
#The plot for temp and atemp are almost identical.
# Humidity and windspeed affect demand but need more statistical analysis.

#Data visualisation - Categorical features vs demand
colors =['g','r','m','b']
cat_list = bikes_prep['season'].unique()
cat_average = bikes_prep.groupby('season').mean()['demand']
plt.subplot(3,3,1)
plt.title('Avg Demand per Season')
plt.bar(cat_list,cat_average,color=colors)

cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
plt.subplot(3,3,2)
plt.title('Avg Demand per Month')
plt.bar(cat_list,cat_average,color=colors)

cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
plt.subplot(3,3,3)
plt.title('Avg Demand per Holiday')
plt.bar(cat_list,cat_average,color=colors)

cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
plt.subplot(3,3,4)
plt.title('Avg Demand per weekday')
plt.bar(cat_list,cat_average,color=colors)

cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
plt.subplot(3,3,5)
plt.title('Avg Demand per hour')
plt.bar(cat_list,cat_average,color=colors)

cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
plt.subplot(3,3,6)
plt.title('Avg Demand per year')
plt.bar(cat_list,cat_average,color=colors)

cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
plt.subplot(3,3,7)
plt.title('Avg Demand per workingday')
plt.bar(cat_list,cat_average,color=colors)

cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
plt.subplot(3,3,8)
plt.title('Avg Demand per Weather')
plt.bar(cat_list,cat_average,color=colors)
plt.tight_layout()

# Notes - Updated
# Demand is not normally distributed.
#Temperature and demand appears to have direct correlation.
#The plot for temp and atemp are almost identical. - check multicollinearity?
# Humidity and windspeed affect demand but need more statistical analysis. check Peasrson correlation coefficient?
# Deamnd by weekday does not change much so it's irrelevant.
# Demand by Year, we have only 2 years data. limited data, better to drop this columns as well.
# Deamnd by workingday does not change much so it's irrelevant.
# demand per hour shows the time when demand for bikes are high and low. 

# check Outliers

bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])


# check the Assumption of Multiple Regression

# Linearity using correlation coefficient matrix using corr
#We should consider the correlation with the dependent variable(demand) and should also check the dependency among the variable. 
correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()

# Notes - Updated
# Demand is not normally distributed.
#Temperature and demand appears to have direct correlation.
#The plot for temp and atemp are almost identical. - check multicollinearity? -> strong correlation among the two, so drop atemp.
# Humidity and windspeed affect demand but need more statistical analysis. check Peasrson correlation coefficient? -> windspeed has a weak correlation with demand. so drop it.
# Demand by weekday does not change much so it's irrelevant. ->drop
# Demand by Year, we have only 2 years data. limited data, better to drop this columns as well. -> drop
# Deamnd by workingday does not change much so it's irrelevant.-> drop.
# demand per hour shows the time when demand for bikes are high and low. 


bikes_prep = bikes_prep.drop(['atemp','windspeed','year','workingday','weekday'],axis=1)


#Test Auto-correlation in Demand using acorr plot
#covert to float as the acorr expects float values
df1 = pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.acorr(df1, maxlags=12)

# Notes - Updated
# Demand is not normally distributed.
#Temperature and demand appears to have direct correlation.
# demand per hour shows the time when demand for bikes are high and low. 
# demand has very high autocorrelation.Autocorrelation exist in dependent varibale so we cannot drop it.

# log- normalise the feature demand
df1 = bikes_prep['demand']
df2 = np.log(df1)

plt.figure()
df1.hist(rwidth=0.9, bins=20)

plt.figure()
df2.hist(rwidth=0.9, bins=20)

bikes_prep['demand'] = np.log(bikes_prep['demand'])

# Notes - Updated
# Demand is not normally distributed. -> log-normalised
#Temperature and demand appears to have direct correlation.
# demand per hour shows the time when demand for bikes are high and low. 
# demand has very high autocorrelation.Autocorrelation exist in dependent varibale so we cannot drop it.

#Autocorrelation
t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']



bikes_prep_lag = pd.concat([bikes_prep, t_1,t_2,t_3],axis=1)

bikes_prep_lag = bikes_prep_lag.dropna()

# Notes - Updated
#Temperature and demand appears to have direct correlation.
# demand per hour shows the time when demand for bikes are high and low. 
# demand has very high autocorrelation.Autocorrelation exist in dependent varibale so we cannot drop it. ->done


# Create dummy variables and drop the first to avoid dummy variables trap for categorical columns/features
# get_dummies drop_first to avoid the problem of multocollinearity
bikes_prep_lag.dtypes
#change the datatype to object/category
bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')
bikes_prep_lag =pd.get_dummies(bikes_prep_lag,drop_first= True)

# Notes - Updated
# Demand is a time dependent or time series data and is not randomly distributed.
# we have to make autocorrelation among the various values in the demand column.
# So we cannot randomly select the data for training and testing.
# Therefore we will have to take complete data for a time period. It could be from start or middle or end.

y = bikes_prep_lag[['demand']] # we do this to get the dataframe and not the series.
x= bikes_prep_lag.drop(['demand'],axis=1)

#split the dataset to train and test.
# create training set at 70%
tr_size = 0.7 * len(x)
tr_size =int(tr_size)

x_train = x.values[0: tr_size]
x_test = x.values[tr_size: len(x)]

y_train = y.values[0: tr_size]
y_test = y.values[tr_size: len(y)]



from sklearn.linear_model import LinearRegression
# train the dataset
std_reg = LinearRegression()
std_reg.fit(x_train, y_train)

# calculate the R-squared values for trainig and test dataset
r2_train = std_reg.score(x_train, y_train)
r2_test = std_reg.score(x_test, y_test)

# Create Y prediction

y_predict = std_reg.predict(x_test)

# import the Mean squared error to calculate the root mean squared error or RMSE
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test,y_predict))


# RMSLE for Kaggle competition

# RMSLE is very similar to RMSE except it takes log values and it adds 1 to actual and predicted values to avoid taking log of any zero values.
# RMSLE is preferred for non-negative predictions and for less vairation for small and large predictions

# calculate the RMSLE
# convert log values in demand to numeric

y_test_e = []
y_predict_e = []

for i in range(0, len(y_test)):
    y_test_e.append(math.exp(y_test[i]))
    y_predict_e.append(math.exp(y_predict[i]))
    
# calculte the sum
log_sq_sum = 0.0
for i in range(0, len(y_test_e)):
    log_a = math.log(y_test_e[i]+1)
    log_p = math.log(y_predict_e[i]+1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff

RMSLE = math.sqrt(log_sq_sum/len(y_test))

print(RMSLE)








