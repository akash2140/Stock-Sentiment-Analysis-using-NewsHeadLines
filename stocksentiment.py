#stock sentiment analysis using news headlines


import pandas as pd

data=pd.read_csv('C:/Users/AKASH KUMAR/Downloads/Stock-Sentiment-Analysis-master/Stock-Sentiment-Analysis-master/Data.csv',encoding = "ISO-8859-1")


print(data.head())

#here class label 1 indicates price increased
#class label 0 indicates price decreased

#classify data into train and test based on date

train=data[data['Date'] < '20150101']
test=data[data['Date']> '20141231']

data1=train.iloc[:,2:27]

#removing all extra character except a-Z and A-Z

data1.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

#print(data1.head())



#renaming column name for ease of understanding to 0-24

list1=[i for i in range(25)]
new_index=[str(i) for i in list1]
data1.columns=new_index

#print(data1.head())


#converting headlines to lower case characters to reduce the case sensitive conflict in bag of words

for index in new_index:
    data1[index]=data1[index].str.lower()

#print(data1.head())


#form statement from all comments to apply NLP easily and to form bag of words

headLines=list()

l=len(data1.index)

for i in range(0,l):
    headLines.append(' '.join(str(x) for x in data1.iloc[i,0:25]))
#print(headLines[0])

#APPLY NLP

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#implement bag of words
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headLines)

#implement random forest classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


#prediction
test_transform=list()

m=len(test.index)

for i in range(0,m):
    test_transform.append(' '.join(str(x) for x in test.iloc[i,2:27]))

test_ds=countvector.transform(test_transform)
predictions = randomclassifier.predict(test_ds)

#checking accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)

print("Accuracy:"+score)

                     






