# https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np

eng_stopwords = stopwords.words('english')
corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
          'I love green eggs, ham, sausages and bacon!',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'    
        ]

labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals', 'weather', 'animals']

corpus = np.array(corpus)
dataset = pd.DataFrame({'Document': corpus,
                        'Category': labels
                        })
# pre-process data
def normalize_Data(doc):
    doc = re.sub('[^a-zA-Z\s]', '', doc)
    doc = doc.lower()
    doc = doc.strip()
    tokens = doc.split()
    filtered = [token for token in tokens if token not in eng_stopwords]
    return ' '.join(filtered)

normalize_corpus = np.vectorize(normalize_Data)
norm_corpus = normalize_corpus(corpus)

# unigram bag of words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus).toarray()
# get all unique words from corpus
vocab = cv.get_feature_names()
cv_df = pd.DataFrame(cv_matrix, columns = vocab)

# n-gram bag of words
bv = CountVectorizer(ngram_range=(2,2))  # you can 1,2 means unigrams as well as bigrams
bv_matrix = bv.fit_transform(norm_corpus).toarray()
bv_vocab = bv.get_feature_names()
bv_df = pd.DataFrame(bv_matrix, columns = bv_vocab)

# tf-idf method
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(norm_corpus).toarray()
tv_vocab = tv.get_feature_names()
tv_df = pd.DataFrame(tv_matrix, columns = tv_vocab)

# to find similary
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)


# predict models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

Y = np.array(labels)
X_train, X_test, Y_train, Y_test = train_test_split(tv_matrix, Y, test_size=0.50, random_state=0)
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
