#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[2]:


df = sns. load_dataset('titanic')


# In[3]:


df= df[['sex','age','survived']]


# In[4]:


df


# In[5]:


sns.boxplot(x='sex',y='age',data=df)


# In[6]:


sns.boxplot(x='sex',y='age',hue='survived', data=df)


# In[ ]:




