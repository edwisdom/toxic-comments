# This script simply outputs some plots (as PNGs) of summary statistics of the dataset.

import sys, os, re, csv, codecs
import pandas as pd 
import numpy as np

import scipy.stats as ss

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import matplotlib_venn as venn 

# Load pre-trained word vectors
EMBEDDING ='data/glove.840B.300d.txt' 
# EMBEDDING='data/crawl-300d-2M.vec'

# Save training and testing data
TRAIN_DATA ='train.csv' 
TEST_DATA ='test.csv'
SAMPLE_SUB ='sample_submission.csv'

# Load data into pandas
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
submission = pd.read_csv(SAMPLE_SUB)

list_train = train["comment_text"].fillna("_na_").values
list_test = test["comment_text"].fillna("_na_").values

# Label comments with no tag as "clean"
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)

# Plot class imbalance
x=train.iloc[:,2:].sum()
plt.figure(figsize=(10,5))
ax= sb.barplot(x.index, x.values)
plt.title("Number of Sentences Per Class in Training Data")
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Class ', fontsize=12)
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

# plt.savefig('imbalance.png')

# Remove the clean column from the dataframe
no_clean=train.iloc[:,2:-1]

col_1 = "toxic"
corr = []
for other_col in no_clean.columns[1:]:
    ct = pd.crosstab(no_clean[col_1], no_clean[other_col])
    corr.append(ct)
cross_tabs = pd.concat(corr, axis=1, keys=no_clean.columns[1:])
table = cross_tabs.to_html()
