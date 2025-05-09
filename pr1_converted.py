import pandas as pd

df=pd.read_csv('Student_performance_data _.csv')
df

df.shape

df.head

df.tail()

df.count()

df.info()

df.isnull()

df.isnull().sum()

df.dropna()

df.fillna(0)

df.describe()

df['Extracurricular'].mean()

df['Music'].mode()

df['Sports'].median()



