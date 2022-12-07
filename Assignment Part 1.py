#!/usr/bin/env python
# coding: utf-8

# ### Predict the price of a house
# #### Steps followed in order to achieve the problem statement:
# ##### 1) Import neccesary libraries
# ##### 2) Understanding Data, Preprocessing and Cleaning
# ##### 3) Exploratory Data Analysis (EDA)
# ##### 4) Model Building 
# ###### Exploration of different models
# ###### - Multiple Linear Regression
# ###### - Polynomial Regression
# ###### - Random Forest Regressor
# ###### - Random Forest Regressor with hyperparameter tuning
# ##### 5) Results and Conclusion
# ***

# #### Import neccesary Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Understanding Data and Preprocessing

# *Import the data*

# In[2]:


df = pd.read_excel("C://Users//honey//Downloads//DS - Assignment Part 1 data set.xlsx")
df.head()


# *Understanding the data*

# In[3]:


df.shape


# `Dataset has 9 columns (features) and 414 rows (records).The 'House price of unit area' column is the target variable and the remaining are the feature variables based on which we will predict the price of a house.`

# In[4]:


df.describe()


# `This provides the summary statistics for all variables in the dataset.`

# In[5]:


df.info()


# `Info gives null / not-null count and Data Types for every column.`
# * `As from the above we can conclude that we have only numerical columns having 'int' and 'float' data types.`
# * `The dataset has no missing values.`

# In[6]:


df.nunique()


# `"nunique" describes number of unique values per column.`

# In[7]:


duplicates = df.duplicated()
duplicates.sum()


# * `The above code removes all the duplicate rows from the dataset.`
# * `Here it doesn't have any duplicate rows.`

# #### Exploratory Data Analysis

# *Outlier Detection and Removal*

# In[30]:


plt.figure(figsize=(13,6))
for feat,i in zip(df,range(0,9)):
    plt.subplot(3,3,i+1)
    sns.boxplot(y=df[feat], color='grey')
    plt.ylabel('Value')
    plt.title('Boxplot\n%s'%feat)
plt.tight_layout()


# `Visualizing boxplot for every column in order to detect outliers.`

# In[31]:


#Finding the 25th percentile and 75th percentiles.
Q1 = df.quantile(0.25)             
Q3 = df.quantile(0.75)

#Inter Quantile Range (75th perentile - 25th percentile)
IQR = Q3 - Q1                          

#Finding lower and upper bounds for all values. All values outside these bounds are outliers
lower_limit=Q1-1.5*IQR                        
upper_limit=Q3+1.5*IQR

per = pd.DataFrame(((df.select_dtypes(include=['float64','int64'])<lower_limit) | (df.select_dtypes(include=['float64','int64'])>upper_limit)).sum(),columns=['Outliers'])
per['Percentage of Outliers'] = [(round((o/414)*100,2)) for o in per.Outliers]
print(per)


# * `Counting Outlier's count and percentage for every column.`
# * `The maximum percentage of Outliers is 8% which is very less.` 

# #### Univariate Analysis 

# *Visualzing and Analyzing every column individually.*

# In[10]:


p=sns.displot(data= df, x='House Age', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `The above graph depicts that the 'House Age' in between 10-20.`

# In[11]:


p=sns.displot(data= df, x='Distance from nearest Metro station (km)', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `We can conclude that houses are majorly 0-1000 km distant from nearest metro station.`

# In[12]:


p=sns.displot(data= df, x='Number of convenience stores', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `The number of convenience stores are variant.`

# In[13]:


p=sns.displot(data= df, x='latitude', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `The above graph says the latitude lies between the range 24.96-24.99 .`

# In[14]:


p=sns.displot(data= df, x='longitude', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `The above graph says the latitude lies between the range 121.53-121.55 .`

# In[15]:


p=sns.displot(data= df, x='Number of bedrooms', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `The number of bedrooms are variant.`

# In[16]:


p=sns.displot(data= df, x='House size (sqft)', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `The House size are majorly ranged between 400-600 sqft and 900-1400 sqft.`

# In[17]:


p=sns.displot(data= df, x='House price of unit area', kde=True , color='green',bins=50)
p.fig.set_dpi(80)


# `The House of price mainly relies between 40-50 unit area.`

# #### Bivariate Analysis 

# *Check for Linearity.*

# In[18]:


plt.figure(figsize=(6,4))
sns.regplot(x=df['House Age'], y=df['House price of unit area']);


# `The above plot depicts that as the house age increases, the price decreases.`

# In[19]:


plt.figure(figsize=(6,4))
sns.regplot(x=df['Distance from nearest Metro station (km)'], y=df['House price of unit area']);


# * `The above plot depicts that as the distance to the nearest MRT station increases, the price decreases. `
# * `This states that the houses that are near to MRT station have higher price.`

# In[21]:


sns.barplot(x=df['Number of convenience stores'], y=df['House price of unit area']);


# `As the number of convenience stores increase in the locality, House price goes up. This shows positive relation between these attributes.`

# In[22]:


sns.lineplot(x=df['House Age'], y=df['Distance from nearest Metro station (km)']);


# `This shows that houses with an average age of 15 - 20 years have high distances to MRT station while the houses aged for 35+ years are more closer to the stations.`

# `From the Bivariate analysis, we can conclude there is some linear relationship but not a perfect one.Therefore we can compare models between Multiple Linear Regression (as data is near to linear) and Polynomial Regression (as data has shown non-linear patterns too).`

# #### Multivariate Analysis 

# In[23]:


plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x=df['Distance from nearest Metro station (km)'], y=df['Number of convenience stores'], hue=df['House price of unit area'])


# `According to the above plot , the House price goes up comparatively as 1) there is a an increase in 'Number of convenience stores' and 2) the distance is minimum i.e between 0-2000 km.`

# In[24]:


plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x=df['latitude'], y=df['longitude'], hue=df['House price of unit area'])


# `According to the above plot , the House price goes up comparatively as 1) the latitude is in the range 24.96-24.98. and 2)the longitude is in the range 121.53-121.55.`

# In[27]:


#doubt
plt.figure(figsize=(5, 5), dpi=100)

sns.scatterplot(data=df, y=df['House price of unit area'], x=df['Transaction date'] , hue= 'House Age', palette="rocket")


# *Creating a pairplot for the dataset.*

# In[25]:


sns.pairplot(df)


# `Pairplot displays multiple pairwise bivariate distributions in a dataset.`

# *Creating a correlation matrix*

# In[28]:


cor = df.corr()
ax = sns.heatmap(cor,annot=True,linewidths=.5)


# `Correlation matrix explains the relationships between every variable.`

# In[29]:


cor_target = abs(cor['House price of unit area'])
cor_target


# `Correlation between all independent variables with the dependent variable.`
# 
# `- "Distance from nearest Metro station (km)" has the highest correaltion with "House price of unit area."`

# In[60]:


cor1 = df.corr()
sns.heatmap(cor1, annot = True,linewidths=.5)


# `Correlation between all independent variables:`
# 
# `- Data doesn't have High correlation amongst features. There is almost low chance of collinearity`
# ***
# 

# *Conclusion from EDA and Graph plots:*

# `- Data is clean having no null values.` 
# 
# `- Data has very less outliers.` 
# 
# `- Data doesn't have High correlation amongst attributes.`
# 
# `- Houses with more convenience stores in the area, with low age have high prices.`
# 
# `- Houses that are aged have more MRT stations near them and fall in low price.`

# ###### Model Building

# *Splitting Data as features and Target variables.*

# In[32]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# *Independent Features*

# In[33]:


X.head()


# *Target Variable*

# In[34]:


y.head()


# *Splitting Data into train and test data.*

# In[65]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)


# *Scaling the data by using Min-Max Scaler.*

# In[38]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)


# `Transform features by scaling each feature  individually such that it is in the given range on the training set, e.g. between zero and one.`

# ###### Multiple Linear Regression

# * *As our problem statement is a regression problem with multiple independent variables, the thought to choose a model will be 'Multiple Linear Regression'.*
# * *The model depicts that how strong the relationship is between two or more independent variables and one dependent variable.*

# *Model building*

# In[66]:


from sklearn.linear_model import LinearRegression
# create instance of the model and storing it to variable linear_model
Linear_model = LinearRegression()
# fit this into xtrain and y train to create the model
Linear_model.fit(X_train,y_train)

# next predict the values in the x test using this model created
# and storing those values to variable y_pred
lr_pred = Linear_model.predict(X_test)

print("Training Score using LR:",Linear_model.score(X_train,y_train))
print("Testing Score using LR:",Linear_model.score(X_test,y_test))


# `Above is the Training and Test score from multiple linear regression.`

# In[82]:


from sklearn import metrics
MAE1 = metrics.mean_absolute_error(y_test, lr_pred)
MSE1 = metrics.mean_squared_error(y_test, lr_pred)
RMSE1 = np.sqrt(MSE1)
r21 = metrics.r2_score(y_test,lr_pred)
print(f"MAE:{MAE1} \nMSE:{MSE1}\nRMSE:{RMSE1} \nR2_Score:{r21}")


# `Above are the MAE,MSE and RMSE scores for multiple linear regression.`

# In[41]:


pd.DataFrame({"Y test":y_test, "Y_pred":lr_pred}).head()


# `Comparision between Actual and predicted.`

# `From the above observations we can conclude that Multiple Linear Regression does not work that well with both training and testing.`

# ###### Polynomial Regression

# * *From EDA, we can conclude that data has shown non-linear patterns too. Also Multiple Linear Regression is not performing well.*
# * *The polynomial models can be used in such a situation where the relationship between study and explanatory variables is curvilinear or sort of non-linear patterns.*

# *Choosing the best polynomial degree through RMSE of train and test*

# In[45]:


# Train List of RMSE per degree
train_RMSE_list=[]
#Test List of RMSE per degree
test_RMSE_list=[]

for d in range(1,10):
    
    #Preprocessing
    #create poly data set for degree (d)
    polynomial_converter= PolynomialFeatures(degree=d, include_bias=False)
    poly_features= polynomial_converter.fit(X)
    poly_features= polynomial_converter.transform(X)
    
    #Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    
    #Train the Model
    polymodel=LinearRegression()
    polymodel.fit(X_train, y_train)
    
    #Predicting on both Train & Test Data
    y_train_pred=polymodel.predict(X_train)
    y_test_pred=polymodel.predict(X_test)
    
    #Evaluating the Model
    
    #RMSE of Train set
    train_RMSE=np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    
    #RMSE of Test Set
    test_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    
    #Append the RMSE to the Train and Test List
    
    train_RMSE_list.append(train_RMSE)
    test_RMSE_list.append(test_RMSE)


# In[46]:


plt.plot(range(1,5), train_RMSE_list[:4], label='Train RMSE')
plt.plot(range(1,5), test_RMSE_list[:4], label='Test RMSE')

plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.legend()


# `Comparision between polynomial degrees between 1 to 10 vs RMSE scores of train and test.`
# * `From the above plot we can conclude that Polynomial Degree of 2 works best with both train and test RMSE scores.`

# *Model building*

# In[42]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
poly = PolynomialFeatures(degree=2, include_bias=True)
x_train_trans = poly.fit_transform(X_train)
x_test_trans = poly.transform(X_test)
Linear_model.fit(x_train_trans, y_train)
y_pred1 = Linear_model.predict(x_test_trans)
print("Training Score using LR:",Linear_model.score(x_train_trans,y_train))
print("Testing Score using LR:",Linear_model.score(x_test_trans,y_test))


# `Above is the Training and Test score from Polynomial Regression with degree 2.`

# In[80]:


from sklearn import metrics
MAE2 = metrics.mean_absolute_error(y_test, y_pred1)
MSE2 = metrics.mean_squared_error(y_test, y_pred1)
RMSE2 = np.sqrt(MSE2)
r22 = metrics.r2_score(y_test,y_pred1)
print(f"MAE:{MAE2} \nMSE:{MSE2}\nRMSE:{RMSE2} \nR2_Score:{r22}")


# `Above are the MAE,MSE and RMSE scores for Polynomial regression.`

# In[44]:


pd.DataFrame({"Y test":y_test, "Y_pred":y_pred1}).head()


# `Comparision between Actual and predicted.`

# `From the above observations we can conclude that Polynomial Regression works pretty well than linear Regression.`

# ###### Random Forest Regressor

# * *A random forest regressor works with data having a numeric or continuous output and they cannot be defined by classes-which is the case in this dataset*
# * *It is an ensemble algorithm which combines multiple random decision trees each trained on a subset of data which gives stability to the algorithm ,improve the predictive accuracy and control over-fitting.*
# 

# In[55]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# *Model building*

# In[72]:


model = RandomForestRegressor()
model.fit(X_train,y_train)
y_pred2=model.predict(X_test)
print("Training Score:",model.score(X_train,y_train))
print("Testing Score:",model.score(X_test,y_test))


# `Above is the Training and Test score from Random Forest Regressor.`

# In[78]:


from sklearn import metrics
MAE3 = metrics.mean_absolute_error(y_test, y_pred2)
MSE3 = metrics.mean_squared_error(y_test, y_pred2)
RMSE3 = np.sqrt(MSE3)
r23 = metrics.r2_score(y_test,y_pred2)
print(f"MAE:{MAE3} \nMSE:{MSE3}\nRMSE:{RMSE3} \nR2_Score:{r23}")


# `Above are the MAE,MSE and RMSE scores for Random Forest Regressor.`

# ###### Random Forest Regressor - Hyperparameter Tuning

# *Hyperparameter tuning aims to find such parameters where the performance of the model is highest or where the model performance is best and the error rate is least.*

# In[49]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = [1.0, 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[50]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# `Above are the 5 hyperparameters that having used for optimizing.`
# * `n_estimators:Number of trees in random forest`
# * `max_features:Number of features to consider at every split`
# * `max_depth:Maximum number of levels in tree`
# * `min_samples_split:Minimum number of samples required to split a node`
# * `min_samples_leaf:Minimum number of samples required at each leaf node`

# *Model building using hyperparameter*

# In[51]:


rf_model = RandomForestRegressor()


# In[52]:


rf_random_model = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid,scoring=None,n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[53]:


rf_random_model.fit(X_train,y_train)
y_predict=rf_random_model.predict(X_test)


# In[54]:


print("Training Score After hyperparamter:",rf_random_model.score(X_train,y_train))
print("Testing Score After hyperparamter:",rf_random_model.score(X_test,y_test))


# `Above is the Training and Test score from Random Forest Regressor using hyperparameter..`

# In[84]:


from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test, y_predict)
MSE = metrics.mean_squared_error(y_test, y_predict)
RMSE = np.sqrt(MSE)
r2 = metrics.r2_score(y_test,y_predict)
print(f"MAE:{MAE} \nMSE:{MSE}\nRMSE:{RMSE} \nR2_Score:{r2}")


# `Above are the MAE,MSE and RMSE scores for Random Forest Regressor using hyperparameter.`

# #### Results and Conclusion 

# *Model comparison*

# In[88]:


results = pd.DataFrame({"Multiple Regression":[MAE1,MSE1,RMSE1,r21],
                       "Polynomial Regression":[MAE2,MSE2,RMSE2,r22],
                       "Random Forest Regressor":[MAE3,MSE3,RMSE3,r23],
                       "RF_Hyperparameter":[MAE,MSE,RMSE,r2]})
results.index=['MAE','MSE','RMSE','R2_Score']
results


# *Conclusion*

# * `The above dataframe shows the comparison between Multiple Regression, Polynomial Regression, Random Forest Regressor and Random Forest with hyperparameter tuning. `
# * `According to results Random Forest Regressor performs quite well amongst all the models.` 
#     * `The MSE and RMSE errors are less in comparison to other models.`  
#     * `The R-square score is higher in RF compared to other models.`
#     * `As higher the R-squared, the better the model fits your data, therefore Random Forest works best for this dataset.`
#     * `The training score for RF was around 94.5 which was quite higher than other models.`
#     * `Although hyperparameter tuning didn't work well in this case.`
# * `Also Polynomial Regression enhanced the results in comparsion to multiple regression as the nature of data was curvilinear (not a perfect linear relationship).` 

# In[ ]:




