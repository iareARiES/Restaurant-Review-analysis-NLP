
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3 )

# Cleaning the texts

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]

for i in range (0,1000):
  review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  words_to_remove = ['not', "isn't"]
  all_stopwords = [word for word in all_stopwords if word not in words_to_remove]
  review = [ps.stem(word)
            for word in review
            if word not in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Naive Bayes model on the Training set

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)

# Predicting the Test set results"""

y_pred = model.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(f"Confusion matrix y_test and y_pred: {cm}")
print(f"Accuracy score : {accuracy_score(y_test,y_pred)}")

cmt = confusion_matrix(y_train, model.predict(X_train))
print(f"Confusion matrix y_train and X_train: {cmt}")
print(f"Accuracy Score: {accuracy_score(y_train, model.predict(X_train))}")