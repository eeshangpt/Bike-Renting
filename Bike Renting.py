
# coding: utf-8

# # Bike Rental Analysis

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from patsy import dmatrices
from statsmodels import api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[2]:


bike_data = pd.read_csv("./day.csv")
bike_data.head()


# In[3]:


bike_data = bike_data.drop(['instant'], axis = 1)
bike_data.head(10)


# In[4]:


bike_data.columns = ['Date', 'Season', 'Year', 'Month', 'Holiday', 'Day_of_week', 'Working_day', 'Weather_situation',
        'Temperature_0', 'Temperature_1', 'Humidity', 'Windspeed', 'Casual_users', 'Registered_users', 'Total_count']


# In[5]:


bike_data.info()


# In[6]:


bike_data.Date = pd.to_datetime(bike_data.Date)


# In[7]:


print('Unique value count for each feature:')
for i in bike_data:
    print(i, '-->', bike_data[i].unique().size)


# In[8]:


bike_data.Season = bike_data.Season.astype('category')
bike_data.Year = bike_data.Year.astype('category')
bike_data.Month = bike_data.Month.astype('category')
bike_data.Holiday = bike_data.Holiday.astype('category')
bike_data.Day_of_week = bike_data.Day_of_week.astype('category')
bike_data.Working_day = bike_data.Working_day.astype('category')
bike_data.Weather_situation = bike_data.Weather_situation.astype('category')


# In[9]:


bike_data.info()


# In[10]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[11]:


plt.bar(bike_data.Season.unique(), np.array(bike_data.groupby(['Season']).count().Date),
        tick_label=['Spring', 'Summer', 'Autumn', 'Winter'], color = '#1A1A1A');


# In[12]:


plt.bar(bike_data.Year.unique(), np.array(bike_data.groupby(['Year']).count().Date),
        tick_label=['2011','2012'], color = '#1A1A1A');


# In[13]:


plt.bar(bike_data.Month.unique(), np.array(bike_data.groupby(['Month']).count().Date), 
        tick_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        color = '#1A1A1A');


# In[14]:


plt.bar(bike_data.Holiday.unique(), np.array(bike_data.groupby(['Holiday']).count().Date), 
        tick_label = ['Not a holiday','Holiday'], color = '#1A1A1A');


# In[15]:


plt.bar(bike_data.Day_of_week.unique(), np.array(bike_data.groupby(['Day_of_week']).count().Date),
        color = '#1A1A1A');


# In[16]:


plt.bar(bike_data.Working_day.unique(), np.array(bike_data.groupby(['Working_day']).count().Date), 
        tick_label = ['Not a workng day', 'Working day'], color = '#1A1A1A');


# In[17]:


plt.bar(bike_data.Weather_situation.unique(), np.array(bike_data.groupby(['Weather_situation']).count().Date),
        tick_label = ['Clear', 'Mist', 'Rain'],
        color = '#1A1A1A');


# In[18]:


plt.hist(bike_data.Temperature_0, bins=40, color = '#1A1A1A');


# In[19]:


plt.hist(bike_data.Temperature_1, bins=40, color = '#1A1A1A');


# In[20]:


plt.hist(bike_data.Humidity, bins = 40, color = '#1A1A1A');


# In[21]:


plt.hist(bike_data.Windspeed, bins=40, color = '#1A1A1A');


# In[22]:


plt.hist(bike_data.Casual_users, bins = 40, color = '#1A1A1A');


# In[23]:


plt.hist(bike_data.Registered_users, bins= 40, color = '#1A1A1A');


# In[24]:


plt.hist(bike_data.Total_count, bins = 40, color = '#1A1A1A');


# In[25]:


numerical = [i for i in bike_data.iloc[:,1:15] if (bike_data[i].dtype == 'float64' or bike_data[i].dtype == 'int64')]
categorical = [i for i in bike_data if not (i in numerical or i == 'Date')]


# In[26]:


def outlier_analysis():
    for i in numerical:
        maxm, minm = box_analysis(i)
        col_type = neutralizeOutlier(maxm, minm, i)
        bike_data[i] = bike_data[i].astype(col_type)


# In[27]:


def box_analysis(i):
    plt.figure()
    print(i)
    sns.boxplot(bike_data[i])
    plt.show()
        
    q1, q3 = np.percentile(bike_data[i], [25,75])
    iqr = q3 - q1
    maxm, minm = q3 + (1.5 * iqr), q1 - (1.5 * iqr)
    return (maxm, minm)


# In[28]:


def neutralizeOutlier(maxm, minm, x):
    max_in = bike_data[bike_data[x] > maxm].index
    min_in = bike_data[bike_data[x] < minm].index
    print("\tOutlier above maximum : ", len(max_in))
    print("\tOutlier below minimum : ", len(min_in), "\n\n")
    
    if bike_data[x].dtype == 'int64':
        maxm, minm = round(maxm), round(minm)
    col_type = bike_data[x].dtype
    
    if len(max_in) > 0:
        for i in max_in:
            bike_data[x].iloc[i] = maxm
    
    if len(min_in) > 0:
        for i in min_in:
            bike_data[x].iloc[i] = minm
    
    return col_type


# In[29]:


outlier_analysis()


# In[30]:


bike_data.info()


# In[31]:


def VIF_calculation(df, i):
    a = df.columns.format()
    a.remove(i)
    feat= "+".join(a)
    
    y, X = dmatrices(i + " ~ " + feat, df, return_type='dataframe')
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["Features"] = X.columns

    return (vif.round(1))


# In[32]:


VIF_calculation(bike_data[numerical], 'Total_count')


# In[33]:


numerical.remove('Temperature_0')


# In[34]:


VIF_calculation(bike_data[numerical], 'Total_count')


# In[35]:


bike_data = bike_data.drop(['Temperature_0', 'Date', 'Casual_users', 'Registered_users'], axis = 1)


# In[36]:


y = bike_data['Total_count'].as_matrix().astype(np.float)
X = bike_data.iloc[:,0:10].as_matrix().astype(np.float)
X = StandardScaler().fit_transform(X)


# In[37]:


def MAPE(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


# In[45]:


def run_algorithm(X, y, X_test, model, model_name):
    reg = model()
    reg.fit(X, y)
    y_new = reg.predict(X_test)
    print(model_name, '\nMean Absolute Percentage Error :{}%\n'.format(MAPE(y_test, y_new)))


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# In[52]:


run_algorithm(X_train, y_train, X_test, LinearRegression, "LINEAR REGRESSION")
run_algorithm(X_train, y_train, X_test, KNeighborsRegressor, "K NEIGHBOUR REGRESSION")
run_algorithm(X_train, y_train, X_test, SVR, "SUPPORT VECTOR REGRESSION")
run_algorithm(X_train, y_train, X_test, RandomForestRegressor, "RANDOM FOREST REGRESSION")
run_algorithm(X_train, y_train, X_test, DecisionTreeRegressor, "DECISION TREE REGRESSION")


# In[74]:


model = RandomForestRegressor(max_depth=10, n_estimators=20)
model.fit(X_train,y_train)


# In[75]:


y_pred = model.predict(X_test)


# In[76]:


print('Mean Absolute Percentage Error :{}%\n'.format(MAPE(y_test, y_pred)))


# In[77]:


output = pd.Series(y_pred)


# In[79]:


output.to_csv("test_output.csv")

