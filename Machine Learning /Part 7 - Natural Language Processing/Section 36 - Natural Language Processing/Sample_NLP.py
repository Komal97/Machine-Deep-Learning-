import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Dataset/Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

#Cleaning dataset

import re
#1. remove all letters except these and removed character will replace by space
review = re.sub('[^A-Za-z]', ' ', dataset['Review'][0])

#2. convert reviews to lowercase
review = review.lower()

#3. remove insignificant words like 'the, a, an etc...'
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#stemming -  taking the root of word
from nltk.stem.porter import PorterStemmer
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)