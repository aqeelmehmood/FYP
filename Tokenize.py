# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:39:54 2020

@author: Aqeel
"""
import pandas as pd
data = pd.read_csv("Comments_45.csv",encoding="latin1",low_memory=False)


#data=data.drop(['Id'], axis = 1)

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0,len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3)
X = cv.fit_transform(corpus).toarray()


# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

#y=pd.get_dummies(data['Label'])
#y=y.iloc[:,:].values

y=data['Label']





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)




from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)
y_pred=model.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)




# Confusion Matrix
from sklearn.metrics import confusion_matrix
CM= confusion_matrix(y_test, y_pred)
import seaborn as sns
import numpy as np
sns.heatmap(CM/np.sum(CM), annot=True,fmt='.2%', cmap='Blues')


#Recall also know as Sensitivity.
# Recall
from sklearn.metrics import recall_score
RC=recall_score(y_test, y_pred, average=None)
print(RC)

#Precission also know as Positive Prediction Power.
# Precision
from sklearn.metrics import precision_score
PS=precision_score(y_test, y_pred, average=None)
print(PS)







