import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Embedding, SpatialDropout1D, Dropout, Activation
from keras.layers import Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Load pre-trained word vectors
EMBEDDING ='data/glove.840B.300d.txt' 
# EMBEDDING='data/crawl-300d-2M.vec'

# Save training and testing data
TRAIN_DATA ='train.csv' 
TEST_DATA ='test.csv'
SAMPLE_SUB ='sample_submission.csv'


embed_size = 300 # Size of word vector, given by our pre-trained vectors
max_features = 150000 # Number of unique words to use (i.e num rows in embedding matrix)
maxlen = 200 # Max number of words in a comment to use


# Load data into pandas
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
submission = pd.read_csv(SAMPLE_SUB)


# Replace missing values in training and test set
list_train = train["comment_text"].fillna("_na_").values
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[classes].values
list_test = test["comment_text"].fillna("_na_").values

# Use Keras preprocessing tools
tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(list_train))
tokenized_train = tok.texts_to_sequences(list_train)
tokenized_test = tok.texts_to_sequences(list_test)

# Pad vectors with 0s for sentences shorter than maxlen
X_t = pad_sequences(tokenized_train, maxlen=maxlen)
X_te = pad_sequences(tokenized_test, maxlen=maxlen)


# Read word vectors into a dictionary
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING))

# Create the embedding matrix
word_index = tok.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# Bidirectional LSTM with 2 convolutional layers, max-pooling, and 1 FC layer
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform", activation="relu")(x)
# avg_pool = GlobalAveragePooling1D()(x)
# x = MaxPooling1D(3)(x)
# x = concatenate([avg_pool, max_pool])
x = Conv1D(64, kernel_size = 6, padding = "valid", kernel_initializer="he_uniform", activation="relu")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=3,
                   verbose=0, mode='auto')
best_model = 'models/model42.h5'
checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

# Fit the model
model.fit(X_t, y, batch_size=1024, epochs=30, callbacks=[es, checkpoint], validation_split=0.1)

# Load best model
model = load_model(best_model)
print ('**Predicting on test set**')
pred = model.predict(X_te, batch_size=1024, verbose=1)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = pred
submission.to_csv('preds/submission16.csv', index=False)

