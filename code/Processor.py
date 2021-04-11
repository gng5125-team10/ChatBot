#Processor.py
import nltk
from wordcloud import WordCloud, STOPWORDS
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import logging

import ChatBotResponses as responses


#for model training, can be removed if we load model from a file
import pandas as pd
import random

books_list = ['bible-kjv.txt', 'edgeworth-parents.txt', 'milton-paradise.txt', 'whitman-leaves.txt', 'austen-sense.txt']

#logger = logging.getLogger('Processor')
#logger.setLevel(logging.DEBUG)


class model_SVM_TFIDF:

  #train model during cration 
  # We can load data from a file in future implementation
  def __init__(self):
    #print (responses.pesponses_sadness)
    self.Train()

  #sklearn.svm should be trained by calling TrainModel before use
  model = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto')) 
  
  #TFIDF vectorizer, should be initialized by calling TrainModel
  tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
     
  #Train model
  def Train(self):
      logging.warning("Started training sklearn.svm model")
      # Create data frame
      #select book samples
      df = selectSamplesOfBooks(300,80)

      #feature TFIDF
      features_tfidf = self.tfidf_vect.fit_transform(df["Content"].values)

      #Train SVM model
      self.model.fit(features_tfidf, df["book_index"].values)

      logging.warning("Finished training sklearn.svm model")
  
  #def loadFromFile(file): 
    #TODO

  #this predicts a book based on text
  def PredictLabel(self, text):
      cleaned_text = prepareText(text)
      feture_vector = self.tfidf_vect.transform([cleaned_text])
      index = self.model.predict(feture_vector)
      return index[0]

      #return index
      #return books_list[index[0]]
  
  def getResponseForALabel(self, label):
      return responses.pesponses_sadness[label]


#this function runs when webserver starts
def Init():
    nltk.download('gutenberg')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')



def lemmatizing(text):
  wn = nltk.WordNetLemmatizer()  
  tokenized_text = text.split()
  words = [wn.lemmatize(word) for word in tokenized_text]
  return ' '.join(words)

def removingStopWords(text):
  tokenized_text = nltk.tokenize.regexp_tokenize(text, "[\w']+")
  stop_words = set(nltk.corpus.stopwords.words('english'))
  words = [w for w in tokenized_text if not w in stop_words]
  return ' '.join(words)

MIN_WORD_LENGTH = 3 
#remove words with length less than this value
def removeShortWords (text, minWordLength = MIN_WORD_LENGTH):
  tokenized_text = text.split()
  words = [token for token in tokenized_text if len(token) > minWordLength-1]
  return ' '.join(words)


#Contractions are shortened version of words such as:
#    "can't": "cannot",
#    "you're": "you are",
#!pip install contractions
import contractions

#convert to lowercase, remove stopwords, special chars, 
def prepareText(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub('[^a-z]', ' ', text) 
    text = removingStopWords(text)
    text = lemmatizing(text)
    text = removeShortWords (text) 
    return text



#for model training, can be removed if we load model from a file
def selectSamplesOfBooks(num_records, sample_length):
  records = []
  labels = []
  enc_labels= []

  for book_indx, each_book in enumerate(books_list):
    text = nltk.corpus.gutenberg.raw(each_book)
    text = prepareText(text)
    tokenized_text = text.split()

    #add label
    label = each_book.lower()
    enc_label = book_indx #use book index

    records_cout = 0
    while records_cout <  num_records:
      #get random sample
      rand = random.randint(55, len(tokenized_text)-150)
      record = ' '.join(tokenized_text[rand:rand+sample_length])
      #delete selected sample
      del tokenized_text[rand:rand+sample_length]

      records.append(record)
      labels.append(each_book)
      enc_labels.append(enc_label)
      records_cout += 1
  
  #create data frame
  data = {'Content': records, 'Label':labels, 'book_index':enc_labels}
  df = pd.DataFrame.from_dict(data)

  return df






