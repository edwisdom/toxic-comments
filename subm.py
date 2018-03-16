import sys, os, re, csv, codecs, numpy as np, pandas as pd

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Dense, Input, LSTM, Embedding, SpatialDropout1D, Dropout, Activation
# from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
# from keras.models import Model, load_model
# from keras import initializers, regularizers, constraints, optimizers, layers
# from keras.callbacks import EarlyStopping, ModelCheckpoint


# EMBEDDING_FILE='glove.840B.300d.txt' # Computed a GloVe embedding from corpus
# TRAIN_DATA_FILE='train.csv' # Training data
# TEST_DATA_FILE='test.csv' # Testing data
# SAMPLE_SUB ='sample_submission.csv'


# embed_size = 300 # Size of word vector
# max_features = 100000 # Number of unique words to use (i.e num rows in embedding vector)
# maxlen = 150 # Max number of words in a comment to use


# # Load data into pandas
# train = pd.read_csv(TRAIN_DATA_FILE)
# test = pd.read_csv(TEST_DATA_FILE)
# submission = pd.read_csv(SAMPLE_SUB)


# # Replace missing values in training and test set
# list_sentences_train = train["comment_text"].fillna("_na_").values
# list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# y = train[list_classes].values
# list_sentences_test = test["comment_text"].fillna("_na_").values

# # Use Keras preprocessing tools
# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(list_sentences_train))
# list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
# list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
# X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
# X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# model = load_model('model32.h5')
# print ('**Predicting on test set**')
# pred = model.predict(X_te, batch_size=1024, verbose=1)
# submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = pred
# submission.to_csv('submission6.csv', index=False)

s1 = pd.read_csv('submission11.csv')
s2 = pd.read_csv('submission14.csv')
s3 = pd.read_csv('submission15.csv')
s4 = pd.read_csv('submission18.csv')

ensemble = s1.copy()
cols = ensemble.columns
cols = cols.tolist()
cols.remove('id')
for i in cols:
	ensemble[i] = (s1[i] + s2[i] + s3[i] + s4[i]) / 4

ensemble.to_csv('ensemble_embeds.csv', index=False)



