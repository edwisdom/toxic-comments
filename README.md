# Toxic Comment Classification Using Deep Learning

This project uses deep learning, specifically long-short term memory (LSTM) units, gated recurrent units (GRU), and convolutional neural networks (CNN) to label comments as toxic, severely toxic, hateful, insulting, obscene, and/or threatening. Each of the models here reaches **over 98.4% accuracy** on cross-validation and the held-out test set.

By improving automated detection of such comments, we can make online spaces more fruitful and welcoming forums for discussion. 

## Getting Started

These instructions will allow you to run this project on your local machine. 

### Download the Data

You can find the pre-trained word vectors below (note that if you download word vectors with fewer dimensions, you will have to change embed_size in the code):
- [GloVe](https://nlp.stanford.edu/projects/glove/ "GloVe Embeddings")
- [FastText](https://fasttext.cc/docs/en/english-vectors.html "FastText Embeddings")

Place these in a *filepath*/data directory. Create a /models directory for saved models and /preds for saved predictions on a held-out test set.

### Install Requirements

Once you have a virtual environment in Python, you can simply install necessary packages with: `pip install requirements.txt`

Note that if you have a GPU, you will also need to install CUDA 9.0. If you don't have a GPU, you should install Tensorflow instead of Tensorflow-GPU. 

### Clone This Repository

```
git clone https://github.com/edwisdom/toxic-comments
```

### Run Models

Either run a model individually with Python (e.g. `python lstm.py`) or run each of them sequentially with:

```
sh ./seq.sh
```

## Background Research

This section covers some of my basic research in deep learning concepts that helped me understand and implement these models.

### Recurrent Neural Networks for NLP

Recurrent neural networks (RNNs) have been shown to produce astonishing results in text generation, sentiment classification, and even part-of-speech tagging, despite their seemingly simple architecture. RNNs are more powerful than regular/vanilla neural networks because they combine an input vector with a *hidden state vector* before outputting a vector, allowing us to encode the sequence of inputs over time that the network has already seen -- for our NLP application, this will be a sequence of words (encoded as vectors), though you can see an example with a sequence of characters below. 

![alt text][rnn]

[rnn]: https://karpathy.github.io/assets/rnn/charseq.jpeg "An example RNN with 4-dimensional input and output layers, and a hidden layer of 3 units (neurons). This diagram shows the activations in the forward pass when the RNN is fed the characters 'hell' as input. The output layer contains confidences the RNN assigns for the next character (vocabulary is h,e,l,o); We want the green numbers to be high and red numbers to be low."

For more on RNNs, read:
- [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [A Beginner's Guide to RNNs](https://deeplearning4j.org/lstm.html)
- [Comparative Study of CNN and RNN for NLP](https://arxiv.org/pdf/1702.01923.pdf)

### Recurrent Convolutional Neural Networks

In [this paper from 2015](http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Recurrent-Convolutional-Neural-Networks-for-Text-Classification.pdf), the authors find that adding a convolutional layer to an RNN outperforms CNNs and RNNs alone for text classification tasks. The authors argue that while the recurrent part captures long-distance connections between the data (e.g. the beginning and end of sentence), the convolutional part captures phrase-level patterns. By adding a max-pooling layer afterward, a CNN can identify the most important words or phrases.

Although the authors don't include any discussion of a stacked CNN (with both small and large window sizes in order to capture both short- and long-distance patterns), other researchers have similarly found RCNNs to be more accurate for a number of classification tasks:

- [Scene Labeling](http://proceedings.mlr.press/v32/pinheiro14.pdf)
- [Object Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf)
- [Video Classification](http://ieeexplore.ieee.org/document/7552971/)

## Data

The data here consists of over 150,000 sentences from Wikipedia's talk page comments. The comments have been labeled by human moderators as toxic, severely toxic, obscene, hateful, insulting, and/or threatening. For more information about how the dataset was collected, see [the original paper](https://arxiv.org/pdf/1610.08914.pdf).

### Class Imbalances

Note that there are a lot more "clean" comments (marked as 0 for all 6 classes of toxicity) than there are toxic. Moreover, "toxic" is by far the most common label. These class imbalances mean that we should be wary of methods like logistic regression, which would favor the majority class. Since we want to detect toxic online comments, misclassifying most comments as clean would defeat the purpose, even though it might improve model accuracy. Read [here](http://www.chioka.in/class-imbalance-problem/) for more on the class imbalance problem.

![alt text](https://github.com/edwisdom/toxic-comments/blob/master/imbalance.png "Class Imbalance in Training Data")

### Multiple Correlated Labels

As the following table shows, there is significant overlap between "toxic" and the other classes. For example, the sentences labeled "severely toxic" are a subset of those labeled "toxic."

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">severe_toxic</th>
      <th colspan="2" halign="left">obscene</th>
      <th colspan="2" halign="left">threat</th>
      <th colspan="2" halign="left">insult</th>
      <th colspan="2" halign="left">identity_hate</th>
    </tr>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>0</th>
      <th>1</th>
      <th>0</th>
      <th>1</th>
      <th>0</th>
      <th>1</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>toxic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>144277</td>
      <td>0</td>
      <td>143754</td>
      <td>523</td>
      <td>144248</td>
      <td>29</td>
      <td>143744</td>
      <td>533</td>
      <td>144174</td>
      <td>103</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13699</td>
      <td>1595</td>
      <td>7368</td>
      <td>7926</td>
      <td>14845</td>
      <td>449</td>
      <td>7950</td>
      <td>7344</td>
      <td>13992</td>
      <td>1302</td>
    </tr>
  </tbody>
</table>

## Model

Here, I will outline some of the major model parameters that I iteratively tweaked, along with the effect on the model's accuracy rate.

### Baseline Model

I began with a simple model:
- Embedding layer with an input of pre-trained 50-dimensional vectors (GloVe 6B.50D)
- Bidirectional LSTM of size 50, with dropout 0.1
- FC layer of size 25, with dropout 0.1
- Output FC layer of size 6 (one per class)

I used a batch size of 32 and the Adam optimizer, which is an alternative to stochastic gradient descent. Each parameter of the network has a separate learning rate, which are continually adapted as the network learns. For more on the Adam optimizer, read [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). 

**Keras Output:**

```
Epoch 2/2
143613/143613 [==============================] - 378s 3ms/step - loss: 0.0447 - acc: 0.9832 - val_loss: 0.0472 - val_acc: 0.9824

```

