#!/usr/bin/env python
# coding: utf-8

# # Price Prediction using Regression

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# #### **Task 1: Import Dataset and create a copy of that dataset**

# In[2]:


#write code here
data = pd.read_csv('data1.csv')
df = data.copy()


# #### **Task 2: Display first five rows** 

# In[3]:


#write code here
df.head()


# #### **Task 3: Drop 'unnamed: 0' column**

# In[4]:


#write code here
df=df.drop(['Unnamed: 0'],axis =1)


# #### **Task 4: Check the number of rows and columns**

# In[5]:


#write code here
df.shape


# #### **Task 5: Check data types of all columns**

# In[6]:


#write code here
df.dtypes


# #### **Task 6: Check summary statistics**

# In[7]:


#write code here
df.describe()


# #### **Task 7: Check summary statistics of all columns, including object dataypes**

# In[8]:


df.describe(include="all")


# **Question: Explain the summary statistics for the above data set**

# Answer:According to the statistic there are total 215909 observations. But statistical measurements for object type data is showing NaN value however for Price the mean value is 56.73 with maximum value 206.8 and minimum with 16.6.
#             

# #### **Task 8: Check null values in dataset**

# In[9]:


#write code here
df.isnull().sum()


# #### **Task 9: Fill the Null values in the 'price' column.**<br>
# 

# In[10]:


#write code here
df['price']=df['price'].fillna(0)


# #### **Task 10: Drop the rows containing Null values in the attributes train_class and fare**

# In[11]:


#write code here
df.dropna(subset=["train_class","fare"], inplace=True)
print(df)


# #### **Task 11: Drop 'insert_date'**

# In[12]:


#write code here
df=df.drop(['insert_date'],axis=1)


# **Check null values again in dataset**

# In[13]:


#write code here
df.isnull().sum()


# #### **Task 12: Plot number of people boarding from different stations**
# 

# In[14]:


#write code here
sns.countplot(x='origin',data=df)
plt.show()


# **Question: What insights do you get from the above plot?**

# **Answer:** According to the plot the people are mostly travelling from Madrid station and least from Ponferrada station. However, Barcelona is second choice followed by Valencia and tham sevilla station.

# #### **Task 13: Plot number of people for the destination stations**
# 

# In[15]:


#write code here
sns.countplot(x='destination',data=df)
plt.show()


# **Question: What insights do you get from the above graph?**

# **Answer:**

# #### **Task 14: Plot different types of train that runs in Spain**
# 

# In[16]:


#write code here
plt.figure(figsize=(15,6))
sns.countplot(x='train_type',data=df)
plt.show()


# **Question: Which train runs the maximum in number as compared to other train types?**

# **Answer:** From the above plot the train AVE runs maximum numbers of time as compared to other train types.
# 

# #### **Task 15: Plot number of trains of different class**
# 

# In[17]:


#write code here
plt.figure(figsize=(15,6))
sns.countplot(x='train_class',data=df)
plt.show()


# **Question: Which the most common train class for traveling among people in general?**

# **Answer:** The most common train class for traveling among people in general is Truista.
# 

# #### **Task 16: Plot number of tickets bought from each category**
# 

# In[18]:


#write code here
plt.figure(figsize=(15,6))
sns.countplot(x='fare',data=df)
plt.show()


# **Question: Which the most common tickets are bought?**

# **Answer:** The most common tickets bought are Promo fare.

# #### **Task 17: Plot distribution of the ticket prices**

# In[19]:


#write code here
plt.figure(figsize=(15,6))
price=df['price']
sns.distplot(price)
plt.show()


# **Question: What readings can you get from the above plot?**

# **Answer:** According to the plot the most tickets are of price less than 50.       

# ###### **Task 18: Show train_class vs price through boxplot**

# In[20]:


#write code here
plt.figure(figsize=(15,6))
sns.boxplot(x='train_class',y='price',data=df)
plt.show()


# **Question: What pricing trends can you find out by looking at the plot above?**

# **Answer:** The price of ticket Turista Plus is maximum as compare to other ticket prices. However, the minimum outlier representing minimum price as compare to other data for Tursits Plus is also present.

# #### **Task 19: Show train_type vs price through boxplot**
# 

# In[21]:


#write code here
plt.figure(figsize=(15,6))
sns.boxplot(x='price',y='train_type',data=df)
plt.show()


# **Question: Which type of trains cost more as compared to others?**

# **Answer:** The train of AVE type cost most as compared to others.

# ## Feature Engineering
# 

# In[22]:


df = df.reset_index()


# **Finding the travel time between the place of origin and destination**<br>
# We need to find out the travel time for each entry which can be obtained from the 'start_date' and 'end_date' column. Also if you see, these columns are in object type therefore datetimeFormat should be defined to perform the necessary operation of getting the required time.

# **Import datetime library**

# In[23]:


#write code here
import datetime


# In[24]:


datetimeFormat = '%Y-%m-%d %H:%M:%S'
def fun(a,b):
    diff = datetime.datetime.strptime(b, datetimeFormat)- datetime.datetime.strptime(a, datetimeFormat)
    return(diff.seconds/3600.0)                  
    


# In[25]:


df['travel_time_in_hrs'] = df.apply(lambda x:fun(x['start_date'],x['end_date']),axis=1) 


# #### **Task 20: Remove redundant features**
# 

# **You need to remove features that are giving the related values as  'travel_time_in_hrs'**<br>
# *Hint: Look for date related columns*

# In[26]:


#write code here
drop_features = ['start_date','end_date']
df.drop(drop_features,axis=1,inplace=True)


# We now need to find out the pricing from 'MADRID' to other destinations. We also need to find out time which each train requires for travelling. 

# ## **Travelling from MADRID to SEVILLA**

# #### Task 21: Findout people travelling from MADRID to SEVILLA

# In[27]:


#write code here
df1 = df[(df.origin=="MADRID") & (df.destination=="SEVILLA")]


# #### Task 22: Make a plot for finding out travelling hours for each train type

# In[31]:


#write code here
sns.countplot(x='travel_time_in_hrs', hue= 'train_type', data=df1)
plt.show()


# #### **Task 23: Show train_type vs price through boxplot**
# 

# In[32]:


#write code here
plt.figure(figsize=(15,6))
sns.boxplot(x='train_type',y='price',data=df1)
plt.show()


# ## **Travelling from MADRID to BARCELONA**
# 

# #### Task 24: Findout people travelling from MADRID to BARCELONA

# In[33]:


#write code here
df1 = df[(df.origin=="MADRID") & (df.destination=="BARCELONA")]


# #### Task 25: Make a plot for finding out travelling hours for each train type

# In[34]:


#write code here
sns.countplot(x='travel_time_in_hrs', hue= 'train_type', data=df1)
plt.show()


# #### **Task 26: Show train_type vs price through boxplot**

# In[35]:


#write code here
plt.figure(figsize=(15,6))
sns.boxplot(x='train_type',y='price',data=df1)
plt.show()


# ## **Travelling from MADRID to VALENCIA**

# #### Task 27: Findout people travelling from MADRID to VALENCIA

# In[36]:


#write code here
df1 = df[(df.origin=="MADRID") & (df.destination=="VALENCIA")]


# #### Task 28: Make a plot for finding out travelling hours for each train type

# In[37]:


#write code here
sns.countplot(x='travel_time_in_hrs', hue= 'train_type', data=df1)
plt.show()


# #### **Task 29: Show train_type vs price through boxplot**

# In[38]:


#write code here
plt.figure(figsize=(15,6))
sns.boxplot(x='train_type',y='price',data=df)
plt.show()


# ## **Travelling from MADRID to PONFERRADA**

# #### Task 30: Findout people travelling from MADRID to PONFERRADA

# In[39]:


#write code here
df1 = df[(df.origin=="MADRID") & (df.destination=="PONFERRADA")]


# #### Task 31: Make a plot for finding out travelling hours for each train type

# In[40]:


#write code here
sns.countplot(x='travel_time_in_hrs', hue= 'train_type', data=df1)
plt.show()


# #### **Task 32: Show train_type vs price through boxplot**

# In[41]:


#write code here
plt.figure(figsize=(15,6))
sns.boxplot(x='train_type',y='price',data=df1)
plt.show()


# # Applying Linear  Regression

# #### Task 33: Import LabelEncoder library from sklearn 

# In[42]:


#write code here
from sklearn.preprocessing import LabelEncoder


# **Data Encoding**

# In[43]:


lab_en = LabelEncoder()
df.iloc[:,1] = lab_en.fit_transform(df.iloc[:,1])
df.iloc[:,2] = lab_en.fit_transform(df.iloc[:,2])
df.iloc[:,3] = lab_en.fit_transform(df.iloc[:,3])
df.iloc[:,5] = lab_en.fit_transform(df.iloc[:,5])
df.iloc[:,6] = lab_en.fit_transform(df.iloc[:,6])


# In[44]:


df.head()


# #### Task 34: Separate the dependant and independant variables

# In[45]:


#write code here
X = df.drop(['price'], axis=1)
Y = df[['price']]


# #### Task 35: Import test_train_split from sklearn

# In[46]:


#write code here
from sklearn.model_selection import train_test_split


# #### Task 36:**Split the data into training and test set**

# In[47]:


#write code here
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.30, random_state=25,shuffle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# #### Task 37: Import LinearRegression library from sklearn

# In[48]:


#write code here
from sklearn.linear_model import LinearRegression


# #### Task 38: Make an object of LinearRegression( ) and train it using the training data set

# In[49]:


#write code here
lr = LinearRegression()


# In[50]:


#write code here
lr.fit(X_train, Y_train)


# #### Task 39: Find out the predictions using test data set.

# In[51]:


#write code here
lr_predict = lr.predict(X_test)


# #### Task 40: Find out the predictions using training data set.

# In[52]:


#write code here
lr_predict_train = lr.predict(X_train)


# #### Task 41: Import r2_score library form sklearn

# In[53]:


#write code here
from sklearn.metrics import r2_score


# #### Task 42: Find out the R2 Score for test data and print it.

# In[54]:


#write code here
lr_r2_test= r2_score(Y_test,lr_predict)


# #### Task 43: Find out the R2 Score for training data and print it.

# In[55]:


lr_r2_train = r2_score(Y_train,lr_predict_train )


# Comaparing training and testing R2 scores

# In[56]:


print('R2 score for Linear Regression Testing Data is: ', lr_r2_train)
print('R2 score for Linear Regression Testing Data is: ', lr_r2_test)


# # Applying Polynomial Regression

# #### Task 44: Import PolynomialFeatures from sklearn

# In[57]:


#write code here
from sklearn.preprocessing import PolynomialFeatures


# #### Task 45: Make and object of default Polynomial Features

# In[58]:


#write code here
poly_reg = PolynomialFeatures()


# #### Task 46: Transform the features to higher degree features.

# In[59]:


#write code here
X_train_poly=poly_reg.fit_transform(X_train)
X_test_poly =poly_reg.fit_transform(X_test)


# #### Task 47: Fit the transformed features to Linear Regression

# In[60]:


#write code here
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)


# #### Task 48: Find the predictions on the data set

# In[61]:


#write code here
y_train_predicted =poly_model.predict(X_train_poly)
y_test_predict = poly_model.predict(X_test_poly)


# #### Task 49: Evaluate R2 score for training data set

# In[62]:


#evaluating the model on training dataset
#write code here
r2_train = r2_score(Y_train, y_train_predicted)


# #### Task 50: Evaluate R2 score for test data set

# In[63]:


# evaluating the model on test dataset
#write code here
r2_test = r2_score(Y_test, y_test_predict)


# Comaparing training and testing R2 scores

# In[64]:


#write code here
print ('The r2 score for training set is: ',r2_train)
print ('The r2 score for testing set is: ',r2_test)


# #### Task 51: Select the best model

# **Question: Which model gives the best result for price prediction? Find out the complexity using R2 score and give your answer.**<br>
# *Hint: Use for loop for finding the best degree and model complexity for polynomial regression model*

# In[65]:


#write code here
r2_train=[]
r2_test=[]
for i in range(1,6):
    poly_reg = PolynomialFeatures()
    
    X_train_poly,X_test_poly = poly_reg.fit_transform(X_train),poly_reg.fit_transform(X_test)
    poly = LinearRegression()
    poly.fit(X_train_poly, Y_train)
   
    y_train_predicted,y_test_predict = poly.predict(X_train_poly),poly.predict(X_test_poly)
    r2_train.append(r2_score(Y_train, y_train_predicted))
    r2_test.append(r2_score(Y_test, y_test_predict))
    
print ('R2 Train', r2_train)
print ('R2 Test', r2_test)


# #### Plotting the model

# In[66]:


plt.figure(figsize=(18,5))
sns.set_context('poster')
plt.subplot(1,2,1)
sns.lineplot(x=list(range(1,6)), y=r2_train, label='Training');
plt.subplot(1,2,2)
sns.lineplot(x=list(range(1,6)), y=r2_test, label='Testing');


# **Answer**

# In[ ]:


According to the value of R square the model is performing good.

