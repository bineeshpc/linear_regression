#%% [markdown]
# # Problem Statement
# The task involves constructing a multiple linear regression model to predict the demand for shared bikes using the provided dataset. The dataset contains various independent variables related to weather, time, and other factors that might influence bike demand. The goal is to understand the significant variables affecting bike demand and create a model to predict it accurately.
# ## Business Goal:
# * Predict the demand for shared bikes based on various independent variables.
# * Assist management in understanding demand dynamics for strategic planning.
# ## Data Preparation:
# * Some numeric columns (e.g., 'weathersit', 'season') with values 1, 2, 3, 4 need conversion into categorical string values as these numeric values do not imply any order.
# * The 'year' column representing the years 2018 and 2019 might appear insignificant but might hold predictive value considering the increasing bike demand annually.
# Model Building:
# * The target variable ('cnt') represents the total number of bike rentals (casual + registered users).
# * Utilize 'cnt' as the target variable and build a multiple linear regression model using other independent variables.
# Model Evaluation:
# * Assess model performance using the R-squared score on the test set to gauge how well the model predicts bike demand.
# * Utilize the provided code snippet to calculate the R-squared score on the test set.

# ## Data Understanding preparation and EDA

#%%
from matplotlib import rcParams
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn.feature_selection as feature_selection
import sklearn.pipeline as pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

import warnings
warnings.filterwarnings('ignore')
# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27

#%% [markdown]


# Dataset characteristics
# day.csv have the following fields:	
# - instant: record index
# - date : date
# - season : season (1:spring, 2:summer, 3:fall, 4:winter)
# - year : year (0: 2018, 1:2019)
# - month : month ( 1 to 12)
# - holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
# - weekday : day of the week
# - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# + weather_code : 
#     - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp : temperature in Celsius
# - atemp: feeling temperature in Celsius
# - humidity: humidity
# - windspeed: wind speed
# - casual: count of casual users
# - registered: count of registered users
# - count: count of total rental bikes including both casual and registered


#%%

filename = "day.csv"

df = pd.read_csv(filename)
df.head(7)

#%%

# rename the variables to something more meaningful for easier understanding

column_names = {'dteday': 'date',
                'yr': 'year',
                'mnth': 'month',
                'weathersit': 'weather_code',
                'hum': 'humidity',
                'cnt': 'count',
                'atemp': 'apparent_temperature'}

df.rename(columns=column_names, inplace=True)
df
# %%
# shape of the dataset
df.shape

#%% [markdown]

# * 730 records and 16 variables
#%%
# info about the dataset
df.info()

#%% [markdown]

# * Every field is numeric except date.
# * date is in object form and needs to be converted to datetime format.


#%%
# descriptive statistics
df.describe()

#%% 

# number of numerical variables
len(df.describe().columns)

#%% [markdown]
# * There are 15 numerical variables.
# * But some of them need to be converted to categorical variables.
#%% [markdown]

# * count is the target variable. count of total rental bikes including both casual and registered
# * instant is the record index and it is irrelevant to the target variable. 
#
#%%

df1 = df.copy()

# covert date to datetime format
df1['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

df1.head()

#%%
# check the weekdays
df2 = df1[['date', 'weekday', 'workingday', 'holiday', 'year', 'month']]
df2['weekdayactual'] = df2.date.dt.weekday
df2

df2.head(14)
# weekday is not actually matching with the date.

#%%

# check the correlation between the count with casual users and registered users
print(df1["registered"].corr(df1["count"]))
print(df1["casual"].corr(df1["count"]))

df3 = df1[["registered", "casual", "count"]]
df3["reg_casual_sum"] = df3["registered"] + df3["casual"]

# check if the sum of registered and casual users is equal to the count
(df3['reg_casual_sum'] == df3['count']).all()

#%%

#%%
column = 'date'
# number of unique values in date column
print(df1[column].nunique())

# minimum date and maximum date
print(df1[column].min(), df1[column].max())


#%%

#%% [markdown]
# * date is converted to datetime format.
# * in the data mapping used is with sunday as 0, weekday in pandas is with monday as 0. After considering this difference weekday still does not match
# * 2018 January 1 is a monday. But in the data it is a Saturday.
# * I strongly feel that the data is not from 2018 2019 since the weekdays are not matching with the dates. But this will not change the analysis.
# * Drop the instant column since it is irrelevant to the target variable.
# * count is the sum of registered and casual users. So we must drop these two columns, otherwise the model will not have any practical use.
# * drop the date column since we have the year, month, weekday, holiday, workingday columns.
#%%

columns_to_drop = ['instant', 'registered', 'casual', 'date']
# drop the instant column
df1.drop(columns_to_drop, inplace=True, axis=1)

df1.head()
#%%

# a nicer view of the desriptive statistics
df1.describe().applymap('{:,.2f}'.format)

#%%

# check for missing values
df1.isnull().sum()

#%% [markdown]

# * There are no missing values in the dataset.

#%%

# find categorical variables
categorical_variables = []
for column in df1.columns:
    if len(df1[column].value_counts()) < 25:
        # print(column)
        categorical_variables.append(column)
        
categorical_variables.sort()
print(categorical_variables, len(categorical_variables))

#%% [markdown]

# * There are 7 categorical variables.

#%%

# find numerical variables

numerical_variables = list(set(df1.columns) - set(categorical_variables))
numerical_variables.sort()
numerical_variables
#%%

for categorical_variable in categorical_variables:
    display(df1[categorical_variable].value_counts())

    
#%% [markdown]
# * There are 4 seasons. 
#   1. spring
#   2. summer
#   3. fall
#   4. winter
# * There are 2 years.
#   0. 2018
#   1. 2019
# * There are 12 months. 1 to 12
# * There are 2 values in holiday.
#    0. not a holiday
#    1. holiday
# * There are 7 values in weekday. 0 to 6
# * There are 2 values in workingday.
# 0. weekend or holiday
# 1. not weekend or holiday
# * There are 4 values in weather_code.
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# * our dataset does not contain the value 4

#%%

categorical_variables
#%%

# Define mapping dictionary
season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}

# Map 'season' column to new categories using map()
df1['season'] = df1['season'].map(season_mapping)

# Define mapping dictionary for 'year' column
year_mapping = {0: 2018, 1: 2019}

# Map 'year' column to new year labels using map()
df1['year'] = df1['year'].map(year_mapping)


# Dictionary mapping numeric months to their names
month_mapping = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Map 'month' column to month names using map()
df1['month'] = df1['month'].map(month_mapping)

# Dictionary mapping binary holiday indicator to labels
holiday_mapping = {0: 'Not a Holiday', 1: 'Holiday'}

# Map 'holiday' column to labels using map()
df1['holiday'] = df1['holiday'].map(holiday_mapping)

# Dictionary mapping numeric weekdays to their names with Sunday as 0
weekday_mapping = {
    0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 
    3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'
}

# Map 'weekday' column to weekday names using map()
df1['weekday'] = df1['weekday'].map(weekday_mapping)

# Dictionary mapping binary working day indicator to labels
working_day_mapping = {0: 'Not a Working Day', 1: 'Working Day'}

# Map 'working_day' column to labels using map()
df1['workingday'] = df1['workingday'].map(working_day_mapping)

# Dictionary mapping weather codes to shortened categories
shortened_weather_mapping = {
    1: 'Clear/Few Clouds',
    2: 'Mist/Cloudy',
    3: 'Light Snow/Rain',
    4: 'Heavy Rain/Snow/Fog'
}

# Map 'weather_code' column to shortened categories using map()
df1['weather_code'] = df1['weather_code'].map(shortened_weather_mapping)
#%%
len(df1.columns)

#%%

# convert categorical variables to categorical type
for categorical_variable in categorical_variables:
    df1[categorical_variable] = pd.Categorical(df1[categorical_variable])
    
#%%
df1.info()

#%% [markdown]

# # 2. Visualising Data 
# **Understanding the data**
# ## Visualising Numerical Data
#%%

sns.pairplot(df1, diag_kind='kde')

#%%

# heatmap of correlation matrix
sns.heatmap(df1.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.show()

#%% [markdown]

# ## Visualising Categorical Data

#%%
categorical_variables
#%%

plt.figure(figsize=(25, 25))

for k in range(1, 8):
    plt.subplot(4, 2, k)
    sns.boxplot(x=categorical_variables[k-1], y='count', data=df1)
    plt.title(categorical_variables[k-1] + ' vs count')
plt.show()

#%%
plt.figure(figsize=(25, 55))
for k in range(1, 8):
    plt.subplot(4, 2, k)
    df_count = df1.groupby(categorical_variables[k-1])['count'].mean().sort_values()
    display(df_count)
    df_count.plot.bar()
    plt.title(categorical_variables[k-1] + 'vs count')
    
plt.show()

#%% [markdown]

# *Count is higher when it is not a holiday
# * Count is higher in May, July, August, September, June in that order. We have good demand starting from april till September. Low demand on October, November, December, January, February.
# * Summer and Fall has higher demand than Spring and Winter. Spring has the lowest demand.
# * clear whether has higher demand than misty weather and rainy weather
# * Demand is the least on sundays.
# * Demand is more on working days
# * 2019 has higher demand than 2018.
#%% [markdown]

# # 3. Data Preparation
# * Create dummy variables for all categorical variables.
#%%

# create dummy variables for all categorical variables
dummies = []
for categorical_variable in categorical_variables:
    dummy = pd.get_dummies(df1[categorical_variable], drop_first=True)
    dummies.append(dummy)
    
df1.drop(categorical_variables, axis=1, inplace=True)
df1 = pd.concat([df1] + dummies, axis=1)

df1.head()

#%%
np.random.seed(0)
df_train, df_test = train_test_split(df1, train_size = 0.7, test_size = 0.3, random_state = 100)

#%%


numerical_variables

#%%
# Apply scaler() to all numerical variables
scaler = MinMaxScaler()
df_train[numerical_variables] = scaler.fit_transform(df_train[numerical_variables])

#%%

df_train.head()

#%%

df_train.describe()

#%%

df_train[numerical_variables].describe()

#%%
# checking the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train[numerical_variables].corr(), annot = True, cmap="YlGnBu")
plt.show()

#%% [markdown]

# * apparent_temparature is highly correlated with count

#%%

plt.figure(figsize=[6,6])
sns.scatterplot(data=df_train, x="apparent_temperature", y="count")
plt.show()

#%% [markdown]

# ### Dividing into X and Y sets for the model building
#%%

# y_train = df_train.pop("count")
# X_train = df_train

y_train = df_train["count"]
X_train = df_train.drop(["count"], axis=1)

#%%

len(X_train.columns)
#%%

# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=15)             # running RFE
rfe = rfe.fit(X_train, y_train)

#%%

rfe_df = pd.DataFrame({"column": X_train.columns,
              "support": rfe.support_,
              "ranking": rfe.ranking_})

rfe_df.sort_values(by="ranking", inplace=True)

rfe_df

#%%

selected_columns = X_train.columns[rfe.support_]
selected_columns

# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[selected_columns]
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())

#%%

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#%%

y_train_count = lm.predict(X_train_rfe)

#%%
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_count), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)

##

#%%
# 1. Linearity Assumption (Using Partial Residual Plots)
sm.graphics.plot_partregress_grid(lm)
plt.show()

#%%

# 2. Normality of Residuals (QQ Plot)
residuals = lm.resid
fig = sm.graphics.qqplot(residuals, line='45')
plt.show()

#%%
# 4. Independence of Residuals (Durbin-Watson Statistic)

durbinWatson = durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {durbinWatson}")
#%%


df_test[numerical_variables] = scaler.transform(df_test[numerical_variables])

y_test = df_test["count"]
X_test = df_test.drop(["count"], axis=1)

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[selected_columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

#%%
# Making predictions
y_pred = lm.predict(X_test_new)

#%%

# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)

#%%

print("plotting y_test and y_pred to understand the spread.")