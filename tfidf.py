#!/usr/bin/env python
import pandas
from sklearn import svm
from sklearn import cross_validation as cv
import numpy as np
import itertools
import re
import math
from sklearn.metrics import accuracy_score

class Word:
    def __init__(self, string):
        self.__string = string
        self.__parents = []
        self.__children = []
        self.__difference = -1 #difference between the word's two tfidf scores
        self.__pos_probs = np.array([float(0)]*82) #store the probabilities that this word is in each position for the positive examples
        self.__neg_probs = np.array([float(0)]*82) #same for the negative examples

    def add_pos_prob(self,i):
        self.__pos_probs[i] +=1
    def add_neg_prob(self,i):
        self.__neg_probs[i] += 1
    def div_pos_probs(self, val):
        self.__pos_probs /= val
    def div_neg_probs(self, val):
        self.__neg_probs /= val
    def get_pos_prob(self, i):
        return self.__pos_probs[i]
    def get_neg_prob(self, i):
        return self.__neg_probs[i]
    def set_difference(self, val):
        self.__difference = val
    def get_difference(self):
        return self.__difference
    def add_parent(self, parent):
        self.__parents.append(parent)
    def get_parents(self):
        return self.__parents
    def add_child(self, child):
        self.__children.append(child)
    def get_children(self):
        return self.__children
    def get_string(self):
        return self.__string

def get_best_child(word):
    '''
    Go through the lineage of a word and find the child with the highest TFIDF difference
    '''
    best_children = []#keep track of the best child in each 'generation' 
    node = word #the word whose children we are looking at
    while len(node.get_children()) != 0: #stop when we run out of 'generations'

        max_dif = float('-inf')
        best_child = None
        for child in node.get_children(): #go through all children and find best one
            if child.get_difference() > max_dif:
                max_dif = child.get_difference() 
                best_child = child
        best_children.append(best_child) #add it to our list
        node = best_child #then set it as the next node to look through
    
    #see if any of the children we found are better than the word we started with
    max_dif = word.get_difference() 
    best_child = word
    for child in best_children: 
        if child.get_difference() > max_dif:
            max_dif = child.get_difference() 
            best_child = child
    return best_child
        
def add_children(word, stop, word_list):
    '''
    Function to find all children of a word and the children of those children, stopping when the word length is longer than 'stop'. All children of a word are added to that word's children list, as well as the overall word list (passed in as word_list)
    '''
    children = []
    if len(word.get_string())<stop:
        for letter in ['A', 'G', 'C', 'T']:
            children.append(word.get_string() + letter)
            if word.get_string() != len(word.get_string()) * word.get_string()[0]:
                children.append(letter + word.get_string())
            else:
                if letter != word.get_string()[0]:
                    children.append(letter + word.get_string())
        for child in children:
            child_word = Word(child)
            word.add_child(child_word)
            if child not in [thing.get_string() for thing in word_list]:
                word_list.append(child_word)
            add_children(child_word, stop, word_list)

def main():
    ## Read in the data sequences ## 
    data = pandas.read_csv('C_elegans_acc_seq.csv')
    data = data[:400] #Only looking at the first 400 sequences so there is a balance of positive and negative examples; the sample data I've been using has 10 times more negative examples, which just makes the SVM always guess negative

    #### Find the keywords for each sequence in our data ###
    words = [] #keep track of the word objects
    #Find all possible combinations of the four letters, right now up to 6 letters long
    combos = ['AA', 'AG', 'AC', 'AT', 'GA', 'GG', 'GC', 'GT', 'CA', 'CG', 'CC', 'CT', 'TA', 'TG', 'TC', 'TT']
    for thing in combos:
        word = Word(thing)
        words.append(word)
        add_children(word, 6, words)

    accuracies = [] #keep track of all accuracies to find the average
    for i in range(50):
        (train, test) = cv.train_test_split(data, test_size = .2) #split into train/test sets

        good_words = [] #keep track of words with a high tfidf difference
        best_words = [] #keep track of the best child of each good word
        pos = train[ train['class'] == 1].reset_index(drop=True) #all positive training examples
        neg = train[ train['class'] == -1].reset_index(drop=True) #all negative training examples

    #### Calculate "TFIDF" scores for each possible word
        differences = [] #keep track of all tfidf differences; will be used to find the cutoff
        for word in words:
            p_count = [] #keep track of how many times word is in each positive sequence
            n_count = [] #keep track of how many times word is in each negative sequence
            p_seq_count = 0 #keep track of how many positive sequences contain the word
            n_seq_count = 0 #keep track of how many negative sequences contain the word
            for seq in pos['seq']:
                p_count.append(seq.count(word.get_string())) #Count number of times word appears in the sequence and add to p_count
                if seq.count(word.get_string()) != 0: #if the word appeared at least once, increment counter
                    p_seq_count +=1
            for seq in neg['seq']:
                n_count.append(seq.count(word.get_string()))
                if seq.count(word.get_string()) != 0: 
                    n_seq_count+=1
            
            p_count = pandas.Series(p_count)
            n_count = pandas.Series(n_count)
            pos_avg = float(p_count.mean()) #avg number of times word is in positive sequences
            neg_avg = float(n_count.mean()) #avg number of times word is in negative sequences
           
            #TF_pos = (avg # times word appears in pos sequences) * (fraction of pos sequences it appeared in), vice versa for TF_neg
            tf_pos = float(pos_avg)*(float(p_seq_count)/len(pos))
            tf_neg = float(neg_avg)*(float(n_seq_count)/len(neg))
            #IDF_pos = ln( (total # neg sequences) / (# of neg sequences where the word appeard an average of more than (.75 * the pos avg) ) )
            idf_pos = math.log(float(len(neg))/((.75*pos_avg < n_count).sum()+.0001))
            idf_neg = math.log(float(len(pos))/((.75*neg_avg < p_count).sum()+.0001))

            positive = tf_pos*idf_pos
            negative = tf_neg*idf_neg
            word.set_difference(abs(positive-negative))
            differences.append(abs(positive-negative))
        
        differences = np.array(differences)
        cutoff = np.percentile(differences, 75)
        for word in words: # "good words" have a tfidf difference above the 75th percentile
            if word.get_difference() > cutoff:
                good_words.append(word)
            
        for word in good_words: #find the best child of each "good word"
            if get_best_child(word) not in best_words:
                best_words.append(get_best_child(word))

    ## Store probability tables for each word ##
        for word in best_words: # go through each "best word"
            for seq in pos['seq']: # go through each pos sequence
                places = re.finditer(word.get_string(), seq) #find where the word is in the seq
                for thing in places: # go through each place the word was found
#                    pos_places.append(thing.span()[0]) #uncomment if only keeping track of first letter of word
                    for j in range(thing.span()[0], thing.span()[1]): #for each index the word occupies, add one to the word's positive probability list, at that index
                        word.add_pos_prob(j)
            for seq in neg['seq']: #do the same for each negative sequence
                places = re.finditer(word.get_string(), seq)
                for thing in places:
#                    neg_places.append(thing.span()[0]) #uncomment if only keeping track of first letter of word
                    for j in range(thing.span()[0], thing.span()[1]):
                        word.add_neg_prob(j)
            #divide every index by the total number of pos/neg sequences, so that each index in the word's pos/neg probability list has the probability that the word occupies that index
            word.div_pos_probs(len(pos))
            word.div_neg_probs(len(neg))
        
        ## Encode train and test data ##
        train_data = [] #store the encoded probabilities for each training sequence
        test_data = [] #store encoded probabilities for each test sequence

        for t in range(len(train)): #go through training examples
            length = len(train['seq'].iloc[t])
            probs = [0] *(2* length) #each sequence gets encoded in an array twice its length; the first half will hold the positive probabilities and the second half will hold the negative probabilities
            for word in best_words:
                places = re.finditer(word.get_string(), train['seq'].iloc[t]) #find the word in the seq
                for thing in places: #for each place the word is found
#                    probs[thing.span()[0]] = word.get_pos_prob(thing.span()[0])
#                    probs[thing.span()[0]+length] = word.get_neg_prob(thing.span()[0])
                    for h in range(thing.span()[0], thing.span()[1]): #for each index the word occupies
                        probs[h] += word.get_pos_prob(h) #add the word's pos prob to that index in the training sequence's encoded array
                        probs[h+length] += word.get_neg_prob(h) #add the word's neg prob that index in the second half of the encoded array
            train_data.append(probs) #add the encoded array to train_data
      
        #do the same array encoding for the test data
        for t in range(len(test)):
            length = len(test['seq'].iloc[t])
            probs = [0] *(2* length)
            for word in best_words:
                places = re.finditer(word.get_string(), test['seq'].iloc[t])
                for thing in places:
#                    probs[thing.span()[0]] = word.get_pos_prob(thing.span()[0])
#                    probs[thing.span()[0]+length] = word.get_neg_prob(thing.span()[0])
                    for h in range(thing.span()[0], thing.span()[1]):
                        probs[h] += word.get_pos_prob(h)
                        probs[h+length] += word.get_neg_prob(h)
            test_data.append(probs)
        
        classifier = svm.SVC(kernel = 'linear', C=2) #use linear SVM

        classifier.fit(train_data, train['class']) #fit to training data (encoded arrays)
        preds = classifier.predict(test_data) #test on test data (encoded arrays)
        acc = accuracy_score(preds, test['class']) 
        print acc
        accuracies.append(acc)
    accuracies = pandas.Series(accuracies)
    print 'average: '
    print accuracies.mean()
    print 'std dev: '
    print accuracies.std()

if __name__ == '__main__':
    main()

    
