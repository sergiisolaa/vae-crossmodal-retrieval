# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:47:01 2020

@author: Sergi Solà (extracted from https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html)
"""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

class VocabularySentences:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0
        self.stop_words = set(nltk.corpus.stopwords.words('english')) 
        self.lemmatizer = WordNetLemmatizer() 

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        tokens_r = [w for w in tokens if not w in self.stop_words] 
        tokens_l = [self.lemmatizer.lemmatize(w) for w in tokens_r]
        for word in tokens_l:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]
    
    def obtain_topK(self, k):
        word_dict = {}
        if k < self.num_words:
            word_dict = {k: v for k, v in sorted(self.word2count.items())}
            j = k + 1
            for i in range(j,self.num_words):
                word = word_dict.key(i)
                idx = self.to_index(word)
                
                self.word2index.pop(word)
                self.word2count.pop(word)
                self.index2word.pop(idx)
                self.num_words -= 1 
    
    def obtain_voc(self, T):
        for i in range(0,self.num_words):
            word = self.index2word[i]
            count = self.word2count[word]
            
            if count < T:
                idx = self. to_index(word)
                
                self.word2index.pop(word)
                self.word2count.pop(word)
                self.index2word.pop(idx)
                self.num_words -= 1 
                
                

class VocabularyTokens:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        if word in self.word2index.keys():
            return self.word2index[word]
        else:
            return -1
    
    def obtain_topK(self, k):
        print('Obtaining only K vocs')
        word_dict = {}
        if k < self.num_words:
            word_dict = {k: v for k, v in sorted(self.word2count.items())}
            nwords = self.num_words
            print(nwords)
            for i in range(k,nwords):
                key_list = list(word_dict.keys())
                word = key_list[i]
                
                idx = self.to_index(word)
                
                self.word2index.pop(word)
                self.word2count.pop(word)
                self.index2word.pop(idx)            
                
                self.num_words -= 1
            
            y = 0
            self.word2index = {}
            self.index2word = {}            
            for x in self.word2count.keys():
                self.word2index[x] = y 
                self.index2word[y] = x 
                y = y + 1
    
    def obtain_voc(self, T):
        print('Filtering words with few occurrences')
        nwords = self.num_words
        for i in range(0,nwords):
            word = self.index2word[i]
            count = self.word2count[word]
            
            if count < T:
                idx = self. to_index(word)
                
                self.word2index.pop(word)
                self.word2count.pop(word)
                self.index2word.pop(idx)
                self.num_words -= 1 
        
        y = 0
        self.word2index = {}
        self.index2word = {}            
        for x in self.word2count.keys():
            self.word2index[x] = y 
            self.index2word[y] = x 
            y = y + 1
        
        