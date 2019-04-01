import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import glob

import codecs

train_path = './data/train/*.txt'
test_path = './data/val/*.txt'


# reference
# https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb
def get_filename():
    train_filenames = glob.glob(train_path)
    test_filenames = glob.glob(test_path)

    return train_filenames, test_filenames


def load_data(train_filenames, test_filenames):
    train = []
    test = []
    for idx, train_filename in enumerate(train_filenames):
        with codecs.open(train_filename, "r", encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                data = line.strip().split('\n')[0]
                train.append([data, int(idx)])

    for idx, test_filename in enumerate(test_filenames):
        with codecs.open(test_filename, "r", encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                data = line.strip().split('\n')[0]
                test.append([data, int(idx)])

    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test


def data_label_split(train, test):
    train = np.asarray(train)
    test = np.array(test)

    train_x = np.asarray(train[:, 0])
    train_y = np.asarray(train[:, 1], dtype=np.int32)

    test_x = np.asarray(test[:, 0])
    test_y = np.asarray(test[:, 1], dtype=np.int32)

    return train_x, train_y, test_x, test_y


# reference
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
def train_validation_split(train_x, train_y):
    from sklearn.model_selection import train_test_split
    x_train, val_x, y_train, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    return x_train, y_train, val_x, val_y


def decompose_as_one_hot(in_char, warning=True):
    if in_char < 128:
        result = in_char  # 67~
    else:
        if warning:
            print('Unhandled character:', chr(in_char), in_char)
        # unknown character
        result = 128 + 1  # for unknown character

    return [result]


def decompose_string_to_one_hot(string, warning=True):
    one_hot_string = []
    for char in string:
        decomposed_char = decompose_as_one_hot(ord(char), warning=warning)
        one_hot_string.extend(decomposed_char)

    return one_hot_string


def data_to_vector(train, test, max_length):
    one_hot_train = np.zeros((len(train), max_length), dtype=np.int32)
    one_hot_test = np.zeros((len(test), max_length), dtype=np.int32)

    for idx, data in enumerate(train):
        length = len(data)
        sentence_vector = decompose_string_to_one_hot(data)
        if length >= max_length:
            length = max_length
            one_hot_train[idx, :length] = np.array(sentence_vector)[:length]
        else:
            one_hot_train[idx, :length] = np.array(sentence_vector)
    one_hot_train = one_hot_train.reshape([len(train), 1, max_length])

    for idx, data in enumerate(test):
        length = len(data)
        sentence_vector = decompose_string_to_one_hot(data)
        if length >= max_length:
            length = max_length
            one_hot_test[idx, :length] = np.array(sentence_vector)[:length]
        else:
            one_hot_test[idx, :length] = np.array(sentence_vector)
    one_hot_test = one_hot_test.reshape([len(test), 1, max_length])

    return one_hot_train, one_hot_test


def label_to_one_hot(train_y, test_y, output_size):
    train_y_one_hot = np.eye(output_size)[train_y]
    test_y_one_hot = np.eye(output_size)[test_y]

    return train_y_one_hot, test_y_one_hot


class Network(object):
    def __init__(self, sess, max_len, batch_size, output_size):
        self.sess = sess
        self.learning_rate = 0.001
        self.max_len = max_len
        self.batch_size = batch_size
        self.hidden_size = 128
        self.output_size = output_size

        self._build_network()

    def _build_network(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.max_len], name='X')
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, self.output_size], name='y')

        with tf.variable_scope('network'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            self.outputs, states = tf.nn.dynamic_rnn(cell, self.X / 129, dtype=tf.float32)

            self.flatten = tf.reshape(self.outputs, [-1, self.hidden_size])

            self.dense_layer = tf.layers.dense(inputs=self.flatten, units=self.output_size, activation=tf.nn.softmax)

        with tf.variable_scope('train'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dense_layer, labels=self.y))
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_accuracy(self, data, label):
        logits = self.sess.run(self.dense_layer, feed_dict={self.X: data})
        y_pred = tf.argmax(logits, axis=1)
        y_pred = self.sess.run(y_pred)
        y_true = self.sess.run(tf.argmax(label, axis=1))

        return y_pred, y_true


def main():
    max_len = 16
    batch_size = 32
    epochs = 10
    output_size = 9

    train_filenames, test_filenames = get_filename()
    train, test = load_data(train_filenames, test_filenames)
    train_x, train_y, test_x, test_y = data_label_split(train, test)
    train_x, test_x = data_to_vector(train_x, test_x, max_length=max_len)
    train_y, test_y = label_to_one_hot(train_y, test_y, output_size)
    train_x_split, train_y_split, val_x_split, val_y_split = train_validation_split(train_x, train_y)

    with tf.Session() as sess:
        net = Network(sess, max_len=max_len, batch_size=batch_size, output_size=output_size)
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for batch_index in range(0, len(train_x_split), batch_size):
                batch_train_x_split, batch_train_y_split = train_x_split[batch_index:batch_index + batch_size], train_y_split[batch_index:batch_index + batch_size]

                _, loss = sess.run([net.train, net.loss], feed_dict={net.X: batch_train_x_split, net.y: batch_train_y_split})
            print('Epoch: ', epoch, 'loss: ', loss, end='')

            y_pred, y_true = net.get_accuracy(val_x_split, val_y_split)
            print(' Validation acc: ', (y_pred == y_true).sum() / len(val_y_split))

        test_y_pred, test_y_true = net.get_accuracy(test_x, test_y)
        print(' Test acc: ', (test_y_pred == test_y_true).sum() / len(test_x))


if __name__ == '__main__':
    main()
