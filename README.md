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
- Pooling layer (average + max concatenated)
- FC layer of size 25, with dropout 0.1
- Output FC layer of size 6 (one per class)

I used a batch size of 32 and the Adam optimizer, which is an alternative to stochastic gradient descent. Each parameter of the network has a separate learning rate, which are continually adapted as the network learns. For more on the Adam optimizer, read [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). 

**Keras Output:**

```
loss: 0.0447 - acc: 0.9832 - val_loss: 0.0472 - val_acc: 0.9824
```

The model had trained for 2 epochs. This output from Keras shows loss and accuracy on the training data used, and the 10% held out for cross-validation. Since the out-of-sample predictions best indicate how well the model generalizes, the val_loss and val_acc will be the measures that I report for future model iterations.

### Batch Size / Epochs

First, I recognized that I could train for multiple epochs. The network eventually overfits if we add too many epochs, so first, we can add a callback to stop early. In this code, if val_loss doesn't improve after 3 epochs, the model stops training. 

```python
es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=3,
                   verbose=0, mode='auto')
```

Moreover, we can save the best model with the following callback:

```python
best_model = 'models/model_filename.h5'
checkpoint = ModelCheckpoint(best_model, 
                             monitor='val_loss', 
                             verbose=0, 
                             save_best_only=True, mode='auto')
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, callbacks=[es, checkpoint], validation_split=0.1)
```

Second, I migrated the network to my GPU by downloading CUDA and Tensorflow-GPU. This allowed me to change the batch size to 1024 and train my network much faster.

**Loss: 0.0454, Accuracy: 0.9832**

### Dropout 

I wanted to check if I could tune the model's performance by increasing dropout before experimenting with the network's architecture. Almost all of these efforts, done alone, actually lowered the model accuracy.

- Embedding Layer Dropout to 0.2 -- Loss: 0.0469, Accuracy: 0.9827
- Final Layer Dropout to 0.3 -- Loss: 0.0482, Accuracy: 0.9831
- LSTM Dropout to 0.3 -- Loss: 0.0473, Accuracy: 0.9827
- Recurrent Dropout to 0.3 -- Loss: 0.0465, Accuracy: 0.9831

These results make some sense in hindsight, since the network size is relatively small. As [this paper found](https://pdfs.semanticscholar.org/3061/db5aab0b3f6070ea0f19f8e76470e44aefa5.pdf), applying dropout in the middle and after LSTM layers tends to worsen performance. This, of course, didn't explain why increasing dropout in the embedding layer (which comes before the LSTM) worsened performance.

As I found in this [paper on CNNs](https://arxiv.org/pdf/1411.4280.pdf), dropping random weights doesn't actually help when there is spatial correlation in the feature maps. Since natural language also exhibits spatial/sequential correlation, spatial dropout would be a much better choice, since it drops out entire feature maps. After adding a spatial dropout of 0.2 before the LSTM layer, the network finally improved.

**Loss: 0.0452, Accuracy: 0.9834**

### Architecture

First, I experimented with a different RNN cell. I simply reconstructed the previous network's architecture, but replaced LSTM cells with GRU cells. GRU layers only have two gates, a reset and update gate -- whereas the update gate encodes the previous inputs ("memory"), the reset gate combines the input with this memory. 

Whereas the LSTM can capture long-distance connections due to its hidden state, this may not be necessary for identifying toxicity, since a comment is likely to be toxic throughout. For more on the difference between GRUs and LSTMs, read [here](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/). For an evaluation of the two on NLP tasks, see [this paper](https://arxiv.org/pdf/1412.3555v1.pdf). 

Surprisingly, the GRU performed comparably to the LSTM without any further tuning.

**Loss: 0.0450, Accuracy: 0.9832**

Second, I used larger pre-trained embedding vectors (from 50 dimensions to 300). Furthermore, I increased the number of words that the model was using for each comment in increments of 50, going up from 100 originally to 300 once model performance stopped improving. This simple change improved performance significantly for the LSTM.

**Loss: 0.0432, Accuracy: 0.09838**

Third, and perhaps most importantly, I added a convolutional layer of size 64, with a window size of 3, in between the recurrent and FC layers for both the LSTM and GRU network. Although I found RCNNs rather late in my model iteration process, I've explained them above in the [Background Research section](https://github.com/edwisdom/toxic-comments#recurrent-convolutional-neural-networks).

**LSTM - Loss: 0.0412, Accuracy: 0.9842**

**GRU  - Loss: 0.0414, Accuracy: 0.9842**

Finally, I decided to stack another convolutional layer of size 64, with window size 6, before the FC layer for both networks. I also tried to add a FC layer of size 64 before the output layer. Both of these slightly improved the model, although the GRU benefitted more from the additional convolution, whereas the LSTM benefitted more from the added FC layer.


| Loss By Model | CNN Layer | FC Layer |
|:-------------:|:---------:|:--------:|
| GRU           | 0.0406    | 0.0408   |
| LSTM          | 0.0411    | 0.0402   |


Ensembled together, the two best-performing networks here reach **98.48% accuracy**.

### Other Things I Learned That Don't Deserve a Whole Section 

- Learning Rate Optimizers: For my data and model, Adam vastly outperformed both Adadelta, Adagrad, and RMSProp. I include a more thorough comparison of the optimizers below, from Suki Lau on Towards Data Science.

![alt text](https://cdn-images-1.medium.com/max/800/1*OjcTfMw6dmOmP4lRE7Ud-A.jpeg)

- Alternative Embeddings: All the figures I present here use the GloVe vectors, but I also tried to use pre-trained FastText vectors of the same size (300D), and the network performed comparably. 

## Future Work

Here are some things that I did not get to tune that would make for interesting results:

1. Using only max-pooling layers vs. using only average-pooling layers vs. using both
2. Initializing different learning rates and setting a decay rate
3. Different activation functions -- Tanh vs. PreLu vs. ReLu
4. More convolutional layers with larger window sizes to capture long-distance connections

## Credits

I would like to thank Prof. Liping Liu, Daniel Dinjian, and Nathan Watts for thinking through problems with me and helping me learn the relevant technologies faster. 

I got the data for this model from a [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), and I was helped greatly by [this exploratory data analysis by Jagan Gupta](https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda).

