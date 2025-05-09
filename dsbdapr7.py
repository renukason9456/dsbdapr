#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk -U')
get_ipython().system(' pip install bs4 -U')


# In[2]:


import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')


# In[3]:


para=('''Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human (natural) languages. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages in a way that is both valuable and meaningful.

NLP involves several tasks including text classification, sentiment analysis, machine translation, named entity recognition (NER), speech recognition, and more. NLP is widely used in applications such as chatbots, voice assistants, translation tools, and information retrieval systems. For example, when you ask your smartphone a question or search for something on Google, NLP is at play in the background to help interpret and process your query.

One of the most important aspects of NLP is text preprocessing. Preprocessing involves several steps to clean and organize the text data before any analysis or modeling can be done. These steps include tokenization, stopword removal, lemmatization, and stemming. By breaking text into smaller chunks (tokens), removing irrelevant words (stopwords), and reducing words to their base forms (lemmatization or stemming), NLP systems can better understand and analyze text data.

With the rise of machine learning and deep learning, NLP has seen significant advances. Pretrained models such as BERT, GPT, and T5 are used to perform tasks like text classification, question answering, and summarization with impressive accuracy. These models are trained on vast amounts of text data, learning patterns in language that allow them to generate human-like text and answer questions with contextually relevant responses.

Despite these advancements, NLP still faces challenges. Ambiguity, cultural context, idiomatic expressions, and domain-specific language can all make it difficult for computers to fully understand human language. However, ongoing research and innovations in the field continue to push the boundaries of what is possible with NLP and AI in general.
''')


# In[4]:


print(para)


# In[5]:


para.split()


# In[6]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


# In[7]:


sent=sent_tokenize(para)


# In[8]:


sent[2]


# In[9]:


words=word_tokenize(para)


# In[10]:


words


# In[11]:


from nltk.corpus import stopwords


# In[12]:


swords=stopwords.words('english')


# In[13]:


swords


# In[14]:


x=[word for word in words if word not in swords]


# In[15]:


x


# In[16]:


from nltk.stem import PorterStemmer


# In[17]:


ps=PorterStemmer()


# In[18]:


ps.stem('working')


# In[19]:


y=[ps.stem(word) for word in x]


# In[20]:


y


# In[21]:


nltk.download('omw-1.4')


# In[22]:


from nltk.stem import WordNetLemmatizer


# In[23]:


wn1=WordNetLemmatizer()


# In[24]:


wn1.lemmatize('working',pos='v')


# In[25]:


print(ps.stem('went'))


# In[26]:


print (wn1.lemmatize('went',pos='v'))


# In[27]:


z=[wn1.lemmatize(word,pos='v') for word in x]


# In[28]:


z


# In[29]:


import string 


# In[30]:


string.punctuation


# In[31]:


t=[word for word in words if word not in string.punctuation]


# In[32]:


t


# In[33]:


from nltk import pos_tag


# In[34]:


pos_tag(t)


# In[35]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[36]:


tfidf=TfidfVectorizer()


# In[37]:


v=tfidf.fit_transform(t)


# In[38]:


v.shape


# In[39]:


import pandas as pd


# In[40]:


pd.DataFrame(v)


# In[ ]:





# In[ ]:





# In[ ]:




