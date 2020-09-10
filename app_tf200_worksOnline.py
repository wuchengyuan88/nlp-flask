#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reference: https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776

from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.random.set_seed(1)
np.random.seed(1)

# loading tokenizer and Keras model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# It can be used to reconstruct the model identically.
#model = tf.keras.models.load_model("h5_model.h5")

#https://github.com/keras-team/keras/issues/2397
# For me the is only solvable if i load the model inside the flask @app.route POST method.
#https://github.com/keras-team/keras/issues/12379

# Text preprocessing
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.


    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text
    return text


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 500
print(r'MAX_SEQUENCE_LENGTH: '+str(MAX_SEQUENCE_LENGTH))
# This is fixed.
EMBEDDING_DIM = 100
# word2vec uses dimension of 300
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#Found 46660 unique tokens.

######## This part is for testing only (with a test string mystr)
#mystr='''
#The current feedback operational amplifiers (CFOAs) are receiving increasing attention as basic building blocks in analog circuit design. This paper gives an overview of the applications of the CFOAs, in particular several new circuits employing the CFOA as the active element are given. These circuits include differential voltage amplifiers, differential integrators, nonideal and ideal inductors, frequency dependent negative resistors and filters. The advantages of using the CFOAs in realizing low sensitivity universal filters with grounded elements will be demonstrated by several new circuits suitable for VLSI implementation. PSPICE simulations using the AD844-CFOA which indicate the frequency limitations of some of the proposed circuits are included.
#'''

#mystr = clean_text(mystr)
#mystr = mystr.replace('\d+','')

# Truncate and pad the input sequences so that they are
# all in the same length for modeling.
#mystr = tokenizer.texts_to_sequences([mystr])
#mystr = pad_sequences(mystr, maxlen=MAX_SEQUENCE_LENGTH)
#print('Shape of data tensor:', mystr.shape)
# Shape of data tensor: (1, 500)

# Generate predictions for samples
#predictions = model.predict(mystr)
#print(predictions)
# Generate arg maxes for predictions
#classes = np.argmax(predictions, axis = 1)
#print(classes)
########


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    model = tf.keras.models.load_model("h5_model.h5")
    print(1)
    if request.method == 'POST':
        message = request.form['message']
        print(2)
        data = [message]
        print(3)
        data = tokenizer.texts_to_sequences(data)
        print(4)
        data = pad_sequences(data,maxlen=MAX_SEQUENCE_LENGTH)
        print(5)
        predictions = model.predict(data)
        print(6)
        my_prediction = np.argmax(predictions, axis = 1)
        print(7)
        my_prediction = my_prediction[0]
        #print(my_prediction)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(host='0.0.0.0')
