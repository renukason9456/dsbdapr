import pandas as pd


df = pd.read_csv('Social_Network_Ads.csv')

df


#input data
x=df[['Age','EstimatedSalary']]
#output data
y=df['Purchased']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

#cross. validation
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=0, test_size=0.2)


x_train

y_train

from sklearn.linear_model import LogisticRegression

import seaborn as sns
sns.countplot(x=y)

y.value_counts()



classifier = LogisticRegression()

classifier.fit(x_train,y_train)

#predication
y_pred = classifier.predict(x_test)

y_train.shape

x_train.shape



x_train.shape

y_pred

y_test

import matplotlib.pyplot as plt

plt.xlabel('Age')
plt.ylabel('Salary')
plt.scatter(x['Age'],x['EstimatedSalary'],c=y)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

pd.DataFrame(x_scaled).describe()

plt.xlabel('Age')
plt.ylabel('Salary')
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=y)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

y_test.value_counts()






from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

new1=[[26,34000]]
new2=[[57,138000]]

classifier.predict(scaler.transform(new1))

classifier.predict(scaler.transform(new2))




