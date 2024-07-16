#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction Useing XGboost

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Read Dataset

# In[2]:


dataset = pd.read_csv(r'E:\Naresh IT Data Science Course\Course Study Material\October Month Course Material\12th Oct\12th, 13th\7.XGBOOST\Churn_Modelling.csv')
dataset


# In[3]:


X = dataset.iloc[:,3:-1].values
X


# In[5]:


y = dataset.iloc[:,-1].values
y


# # Encoding categorical data
# 
# Label Encoding the "Gender" column

# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])


# In[7]:


print(X)


# ## One Hot Encoding the "Geography" column

# In[9]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))


# In[10]:


print(X)


# In[11]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))


# In[12]:


print(X)


# ## Splitting the dataset into the Training set and Test set

# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# ## Training XGBoost on the Training set

# In[14]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)


# ## Predicting the Test set results

# In[15]:


y_pred = model.predict(X_test)


# In[16]:


y_pred


# ## Making the Confusion Matrix¶

# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# ## Compute ROC curve and AUC

# In[25]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,threshold = roc_curve(y_test,y_pred)
roc_auc = auc (fpr,tpr)
fpr
roc_auc


# ## compute accuracy

# In[21]:


from sklearn.metrics import accuracy_score
Ac = accuracy_score (y_test,y_pred)
print(Ac)


# # # compute Bias and Varince

# In[26]:


Bias = model.score(X_train,y_train)
print(Bias)


# In[28]:


Variance = model.score(X_test,y_test)
print(Variance)


# ## Applying k-Fold Cross Validation

# In[31]:


from sklearn.model_selection import cross_val_score
accuracies  = cross_val_score (estimator = model, X = X_train, y=y_train, cv =10)
print("Accurscies: {:.2f}%".format(accuracies.mean()*100))
print('Standar Deviation : {:.2f}%'.format(accuracies.std()*100))

