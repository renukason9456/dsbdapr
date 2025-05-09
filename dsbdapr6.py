#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Iris.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


#input data
x=df.drop('Species',axis=1)
#output data
y=df['Species']


# In[6]:


y.value_counts()


# In[7]:


#cross validation
from sklearn.model_selection import train_test_split


# In[8]:


x_train ,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)


# In[9]:


x_train.shape


# In[10]:


x_test.shape


# In[11]:


#import the class
from sklearn.naive_bayes import GaussianNB


# In[12]:


#create the object
clf= GaussianNB()


# In[13]:


#train the algorithm
clf.fit(x_train,y_train)


# In[14]:


y_pred=clf.predict(x_test)


# In[15]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[16]:


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


# In[17]:


confusion_matrix(y_test,y_pred)


# In[18]:


from sklearn.metrics import ConfusionMatrixDisplay




# In[19]:


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[20]:


accuracy_score(y_test,y_pred)


# In[21]:


clf.predict_proba(x_test)


# In[22]:


newl = [[4.5, 2.9, 3.1, 0.4, 1.2]]  # ➕ Add value for the 5th feature

clf.predict(newl)[0]


# In[23]:


newl = [[5.5, 3.1, 1.0, 0.8, 2.3]]  # Add the correct 5th value

clf.predict(newl)[0]


# In[24]:


newl = [[6.5, 3.3, 4.9, 1.8, 0.5]]  # <-- now 5 features

clf.predict(newl)[0]


# In[25]:


print(classification_report(y_test,y_pred))


# In[ ]:




