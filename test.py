import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers

model = load_model('model3.h5')

EMBEDDING_FILE='glove.6B.50d.txt' # Computed a GloVe embedding from corpus
TRAIN_DATA_FILE='train.csv' # Training data
TEST_DATA_FILE='test.csv' # Testing data

embed_size = 50 # Size of word vector
max_features = 20000 # Number of unique words to use (i.e num rows in embedding vector)
maxlen = 100 # Max number of words in a comment to use


# Load data into pandas
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)


# Replace missing values in training and test set
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

# Use Keras preprocessing tools
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

y_train = model.predict([X_t], batch_size=1024, verbose=1)

toxic_i = np.argsort(y_train[:,-1])
for i in toxic_i: print(list_sentences_train[i])