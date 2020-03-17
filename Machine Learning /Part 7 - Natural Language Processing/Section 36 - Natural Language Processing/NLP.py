import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv("Dataset/Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

corpus = []

for i in range(0, 1000):
    review = re.sub('[^A-Za-z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer 

#max_feature = reduce sparcity by setting max features which excludes non-frequent used word
cv = CountVectorizer(max_features = 1500) 

#X(independet variable) = feature matrix which is a sparse matrix where columns = words, rows = reviews
X = cv.fit_transform(corpus).toarray()  
#Y = Dependent variable which is like column
Y = dataset.iloc[:,1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#Fitting the naive bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
#55 - correct results for negative reviews
#12 - wrong results for negative reviews
#42 - wrong results for positive reviews 
#91 - correct results for positive reviews

accuracy = (55+91)/200 #200 is total number of reviews