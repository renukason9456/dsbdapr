import numpy as np

X= np.array([2,4,6,8,10,12])
np.mean(X)

np.median(X)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X= np.array([2,4,6,8,10,12])
df= pd.DataFrame(X)
print (df)

plt.boxplot(X)

df.plot.box()

df.plot.hist()

data = {
    "Name": ["Amit", "Renuka", "Raj", "Shital", "Vikram", "Ananya", "Rohan"],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male"],
    "Marks": [85, 80, 78, np.nan, 76, 82, np.nan],
    "Age": [np.nan,21,22,np.nan,24,np.nan,26]
}
df = pd.DataFrame(data)
print(df)

df.head()

df.tail()

df.count()

df.isnull()

df.isnull().sum()

df.dropna()

df.fillna(0)

df['Marks'].fillna(df['Marks'].mean())

df['Age'].fillna(df['Age'].median())

df.fillna(method='bfill')

df.fillna(method='pad')

df['Age'].plot.hist()



