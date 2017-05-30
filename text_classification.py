from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import util

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders
from tensorflow.contrib.session_bundle import exporter
import time

learn = tf.contrib.learn

FLAGS = None

LOG_PATH = 'data/tensorflow/log.log'
FILE_PATH = 'data/tensorflow/data.csv'
FILE_PATH_TEST = 'data/tensorflow/data_test.csv'
MODEL_PATH = 'data/tensorflow/model'

MAX_STEPS = 5000
MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
n_words = 0

# bag_of_words_model OR rnn_model
ALGO = 'bag_of_words_model'


def bag_of_words_model(features, target):
    """A bag-of-words model. Note it disregards the word order in the text."""
    target = tf.one_hot(target, 15, 1, 0)
    features = encoders.bow_encoder(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    logits = tf.contrib.layers.fully_connected(features, 15, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)
    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


def rnn_model(features, target):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    target = tf.one_hot(target, 15, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)

    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


def load_dataset():
    if not os.path.isfile(FILE_PATH) or not os.path.isfile(FILE_PATH_TEST):
        util.create_train_csv(FILE_PATH, limit=10000, target='category')
        util.create_test_csv(FILE_PATH_TEST, limit=100, offset=10000, target='category')

    train_path = FILE_PATH
    test_path = FILE_PATH_TEST

    train = learn.datasets.base.load_csv_without_header(
        train_path, target_dtype=np.int32, features_dtype=np.str, target_column=0)
    test = learn.datasets.base.load_csv_without_header(
        test_path, target_dtype=np.int32, features_dtype=np.str, target_column=0)

    return learn.datasets.base.Datasets(train=train, validation=None, test=test)


def main(unused_argv):
    global n_words
    # Prepare training and testing data
    start = time.time()
    dataset = load_dataset()
    tf.logging.set_verbosity(tf.logging.INFO)

    x_train = pandas.DataFrame(dataset.train.data)[0]
    y_train = pandas.Series(dataset.train.target)
    x_test = pandas.DataFrame(dataset.test.data)[0]
    y_test = pandas.Series(dataset.test.target)

    # Process vocabulary
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    n_words = len(vocab_processor.vocabulary_)

    load_dataset_time = time.time() - start

    # Build model
    # Switch between rnn_model and bag_of_words_model to test different models.
    if ALGO == 'rnn_model':
        model_fn = rnn_model
    else:
        model_fn = bag_of_words_model

    # Train
    start = time.time()
    classifier = learn.Estimator(model_fn=model_fn, model_dir=MODEL_PATH)
    classifier.fit(x_train, y_train, max_steps=MAX_STEPS)
    classifier.evaluate(x_train, y_train, steps=1)
    train_time = time.time() - start

    # predict
    start = time.time()
    y_predicted = [
        p['class'] for p in classifier.predict(
            x_test, as_iterable=True)
        ]
    predict_time = time.time() - start
    score = metrics.accuracy_score(y_test, y_predicted)

    print("Load Dataset time: %.3f sec" % load_dataset_time)
    print("Train time: %.3f sec" % train_time)
    print("Predict time: %.3f sec" % predict_time)

    print('Total words: %d' % n_words)
    print('Accuracy: {0:f}'.format(score))

if __name__ == '__main__':
    tf.app.run(main=main)
