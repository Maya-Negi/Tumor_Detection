#!/usr/bin/env python
# coding: utf-8

# In[36]:


#importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[37]:


#importing and Reading the csv file
df=pd.read_csv("D:\data science\Tumor_Detection.csv")


# In[38]:


df.head()


# In[39]:


df.columns


# In[40]:


df.info()


# In[41]:


df.drop('id',axis = 1, inplace = True)


# In[42]:


df.columns


# In[43]:


type(df.columns)


# In[44]:


l=list(df.columns)
print(l)


# In[45]:


#start point
features_means = l [1:11]
features_se = l [11:21]
features_worst = l [21:]


# In[46]:


print(features_means)


# In[47]:


print(features_se)


# In[48]:


print(features_worst)


# In[49]:


df.head()


# In[50]:


df


# In[51]:


df['diagnosis'].unique()


# In[52]:


x = df["diagnosis"]
sns.countplot(x=x, label='count')


# In[53]:


df.shape


# In[54]:


df.describe()


# In[55]:


# corr = df.corr()  -----ERROR


# In[56]:


# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()


# In[57]:


corr_matrix


# In[58]:


#heatmao
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix)


# In[59]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[60]:


df['diagnosis'].unique()


# In[61]:


x=df.drop('diagnosis', axis=1)
x.head()


# In[62]:


y=df['diagnosis']
y


# In[63]:


# Divide the dataset into train and test set  
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
# from sklearn.preprocessing import standardScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[64]:


df.shape


# In[65]:


x_train.shape


# In[66]:


x_test.shape


# In[67]:


y_train.shape


# In[68]:


x_test.shape


# In[69]:


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)
x_train


# In[70]:


#appply random forest classifier
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:




