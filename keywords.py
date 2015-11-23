#!/usr/bin/env python

import pandas
from sklearn import svm
from sklearn import cross_validation as cv
from sklearn import metrics
import numpy as np
import itertools
import re
import math


def my_kernel(X, Y): #when training, X and Y are both indices for training set; when testing, X is the list of indices for test set and Y is the list of indices for training examples
    R = np.zeros((len(X), len(Y))) #initialize a zero matrix
    for x in X: 
        for y in Y:
            i = int(x[0]) #turn indices into integers
            k=i
            j = int(y[0])
            if (i>= len(X)): #if X is representing the test set, the indices will be offset from zero by the length of the training set; so reverse the offset to make the matrix row indices start at zero
                k = i-len(Y)
            counter = 0
            #count the number of keywords that are in both examples
            for word in data['keywords'][i]:
                if word in data['keywords'][j]:
                    counter +=1
            R[k, j] = counter #matrix entry for [k, j] is how 'similar' examples k and j are
    return R

def main():
    ## Read in the data sequences ## 
    global data 
    data = pandas.read_csv('C_elegans_acc_seq.csv')
    data = data[:400] #Only looking at the first 400 sequences so there is a balance of positive and negative examples; the sample data I've been using has 10 times more negative examples, which just makes the SVM always guess negative
    data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True) #shuffle the data and reset the indices


    #### Find the keywords for each sequence in our data ###
    possible_words = []
    letters = "AGCT"
    #Find all possible combinations of the four letters, right now only 2 and 3 letter combos
    for j in range(2, 4):
        words = [''.join(i) for i in itertools.product(letters, repeat=j)]
        for thing in words:
            possible_words.append(thing)

    new_col = [] #This new column will be added to the data set; it will contain a list of each sequence's keywords

    for k in range(len(data)): #go through each sequence in the data
        keywords = [] #a list to keep track of that sequence's keywords
        for word in possible_words: #go through each possible word
            positions = [] 
            for match in re.finditer(word, (data['seq'].iloc[k])):
                positions.append(match.span()) #if the word is in the sequence, add its indices to positions
            distances = []
            if len(positions)>2: #only look at words that occurred more than two times
                for t in range(len(positions)-1):
                    distances.append(positions[t+1][0]-positions[t][1]+1) #find the distance between each set of indices
                distances = pandas.Series(distances) #make into pandas series to easily calculate mean and standard deviation 
                ##Calculate C using formulas from paper
                normalized_distances = distances/distances.mean()
                sigma = normalized_distances.std() #standard deviation of normalized distances

                n = len(positions) # number of occurrences of word
                p = n/len(data['seq'].iloc[k]) #divide by number of letters in the sequence

                sigma_nor = sigma/(math.sqrt(1-p))
                
                sigma_nor_mean = (2*n-1)/(2*n+2)
                sigma_std_dev = 1/(math.sqrt(n)*(1+2.8*pow(n, -.865)))
                C = (sigma_nor - sigma_nor_mean*n)/(sigma_std_dev*n)

                if (C>1): #Right now just picking 1 as a cutoff value
                    keywords.append(word)

        new_col.append(keywords)

    new_col = pandas.Series(new_col)
    data['keywords'] = new_col #add the column of keywords to the data frame


    ## Training/testing the SVM ##
    k = int(round(.8*len(data))) #use 80 percent of data for training, the rest for testing
    train = data[:k]
    test = data[k:]

    #custom kernel won't accept string data, so pass it arrays of indices:
    X = np.arange(len(train)).reshape(-1, 1) #gives an array of indices for the training set
    Y = np.arange(len(train), len(data)).reshape(-1, 1) #array of indices for test set

    classifier = svm.SVC(kernel = my_kernel)
    classifier.fit(X, train['class']) 
    
    preds = classifier.predict(Y)
    print preds
    accuracy = metrics.accuracy_score(preds, test['class'])
    print accuracy

if __name__ == '__main__':
    main()

    
