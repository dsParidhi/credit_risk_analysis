
# coding: utf-8

# In[1]:


# --------------------BUSINESS SCENERIO-----------------------------
#LGD stands for Loss given default
# so it means when a customer at a bank defaults on his loan
# how muhc money does the bak lose. The customer might have paid
#some amount back or no amount at all. 
# the bank wants to know if the amount the bank loses can be predicted
# for new customers who apply for a loan
# from the past data of all defaulters and their pending amounts


# In[2]:


#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # this is for visualization
import seaborn as sns # for visualization
get_ipython().magic('matplotlib inline')
import statsmodels.formula.api as sm
import scipy, scipy.stats
import math # log function is inside this library
# this is used to get the plots inline i.e. in the same page
from collections import Counter


# In[5]:


# read the csv file into a dataframe
df=pd.read_csv("C:/Users/admin/Desktop/data/LGD_DATA.csv")
#give the location as on your system


# In[6]:


# lets do a basic EDA ( exploratory data analysis) on the file
df.info() # this tells us 15290 rows & 7 columns
#df.shape will also give the same info
#df.info also tells us there are no NULL values


# In[7]:


df.head()


# In[10]:


import seaborn as sns
#to plot histograms
sns.distplot(df['Losses in Thousands'],kde=False,bins=50)


# In[16]:


# this probably means we shud take the log to normalize the data
sns.distplot((df['Losses in Thousands']),kde=False,bins=50)
# map function is used to apply any function on each element of a series/list
# now the distribution looks normal


# In[17]:


sns.distplot(df['Age'],kde=False,bins=50)


# In[18]:


sns.distplot(df['Years of Experience'])


# In[19]:


sns.boxplot()
sns.boxplot(x="Married",y="Losses in Thousands",data=df,hue="Gender")
#this shows that sinle ppl & Male are the worst


# In[20]:


df.corr()
# we see a high co-relatin between Age and Years of Experience 
# which is obvious as with Age your Experience increases


# In[21]:


# first lets build a simple model with all variables and as is
dummy_var=pd.get_dummies(df['Gender'],drop_first=True)

print( type(dummy_var.head()))










# In[22]:


# we will have to convert the string variables to dummy variables ( Do u miss R?)
dummy_var1=pd.get_dummies(df['Gender'],drop_first=True)
#simillarly for married
dummy_var2=pd.get_dummies(df['Married'],drop_first=True)
dummy_var2.head()
# merge the above 2 dataframe with the original dataframe df
df_new=pd.concat([df,dummy_var1,dummy_var2],axis=1)
df_new.head()


# In[23]:


#now we no longer need Married and gender Columns. We will use their
#dummies instead
df_new2=df_new.drop(['Gender','Married'],axis=1)
df_new2.head()


# In[30]:


#decide your regressor and predictor variables
x=df_new2[["Age","Number of Vehicles","M","Single"]]
y=df_new2["Losses in Thousands"]
#split the data into train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression # import the functions
lm=LinearRegression() #call the function
lm.fit(x_train,y_train)# fit the model
print(lm.intercept_) # see the intercept
print(lm.coef_)# see the betas
#see the error parameters
from sklearn import metrics
#make prediction
pred=lm.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, pred)
print (metrics.mean_absolute_error(y_test,pred))


# In[31]:


#so the linear regression equation we get is
#----loss in thousands=539.65-6.14*Age-1.79*Number of Vehicles+97*M+136*Single
#so according to above--
#Young people (- coeff)and Male(+ coeff) & Unmarried person(+coeff)
#dont pay their loans back


# In[32]:


#lets get the P values of each predictors 
from statsmodels.api import add_constant
X2 = add_constant(x_train)
lm= sm.OLS(y_train,X2)
lm2=lm.fit()
lm2.pvalues
# sm.OLS by default does not add an intercept in the model.
#so we manually added it by the first line


# In[33]:


#to see the summary just like in R


# In[34]:


print(lm2.summary())


# In[35]:


#Let us try to improve the model
#we will try the following
# take log of the dependent variable
# create a new variable AGECATEGORY which is "Young" if age<28, "MiddleAged" if 28<age<58
#and "old" if Age>58


# In[57]:


print (df_new2.head())
df_new2["AgeCategory"]=["Young" if df_new2['Age'][i]<=28 else "MiddleAged" if 28<df_new2['Age'][i]<58 else "Old" for i in range(len(df_new2))]
df_new2['Logy']=(df_new2['Losses in Thousands'])
# map(anyfunction, list)--> to apply the function on each element in the list
Counter(df_new2['AgeCategory'])
# we wil have to convert the AgeCategory to dummy varables
dummy_var3=pd.get_dummies(df_new2['AgeCategory'],drop_first=True)
df_new3=pd.concat([df_new2,dummy_var3],axis=1)
df_new3.head()


# In[60]:


#deicde the new x and y variables
x=df_new3[["Age","Number of Vehicles","M","Single","Old","Young"]]
y=df_new3["Logy"]


# In[62]:



from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
lm=LinearRegression() #call the function
lm.fit(x_train,y_train)# fit the model


# In[63]:


#see the error parameters
from sklearn import metrics
#make prediction
pred=lm.predict(x_test)
metrics.mean_absolute_error(y_test,pred)
from sklearn.metrics import r2_score
print (r2_score(y_test, pred))
# the r2 has improved
print (metrics.mean_absolute_error(y_test,pred))


# In[64]:


print(lm.intercept_) # see the intercept
print(lm.coef_)# see the betas
# now the equation will be
#"Age","Number of Vehicles","M","Single","Old","Young"
#log(losses in Thousands)= 5.91-0.005*(Age)+0.005*(#ofVehicles)+
#0.21*(Male)+0.31*(Single)-0.70*(Old)+0.02*(Young)

