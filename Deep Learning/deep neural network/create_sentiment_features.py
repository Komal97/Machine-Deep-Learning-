import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import random
import pickle

lemmatizer = WordNetLemmatizer()
lines = 10000000

def create_lexicon(pos, neg): #pos = positive, neg = negative
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for line in contents[:lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    word_counts = Counter(lexicon)
    
    l2 = []
    for w in word_counts:
        if 1000 > word_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2
    
def sample_handling(sample, lexicon, classification):
    featureset = []
    
    with open(sample,'r') as f:
        contents = f.readline()
        for line in contents[:lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size = 0.01):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos,lexicon, [1,0])
    features += sample_handling(neg,lexicon, [0,1])
    random.shuffle(features)
    
    features = np.array(features)
    
    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y
    

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)