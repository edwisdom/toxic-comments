# Toxic Comment Classification Using Deep Learning

This project uses deep learning, specifically long-short term memory (LSTM) units, gated recurrent units (GRU), and convolutional neural networks (CNN) to label comments as toxic, severely toxic, hateful, insulting, obscene, and/or threatening. By improving automated detection of such comments, we can make online spaces more fruitful and welcoming forums for discussion. 

## Getting Started

These instructions will allow you to run this project on your local machine. 

### Download the Data

You can find the pre-trained word vectors below (note that if you download word vectors with fewer dimensions, you will have to change embed_size in the code):
- [GloVe](https://nlp.stanford.edu/projects/glove/ "GloVe Embeddings")
- [FastText](https://fasttext.cc/docs/en/english-vectors.html "FastText Embeddings")

Place these in a *your_filepath*/data directory. Also create a /models and /preds directory.

### Install Requirements

Once you have a virtual environment in Python, you can simply install necessary packages with: `pip install requirements.txt`

Note that if you have a GPU, you will also need to install CUDA 9.0. If you don't have a GPU, you should install Tensorflow instead of Tensorflow-GPU. 

### Clone This Repository

```
git clone https://github.com/edwisdom/toxic-comments
```

## Past Work

