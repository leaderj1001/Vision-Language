import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, sess, input_size, max_len, char_size, embedding_size):
        self.sess = sess
        self.input_size = input_size

        self.max_len = max_len
        self.char_size = char_size
        self.embedding_size = embedding_size
        self.hidden_size = 256

        self.weights_init = tf.contrib.layers.xavier_initializer()

        self._build_image_network()
        self._build_language_network()

    def _build_image_network(self):
        self.image_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size[0], self.input_size[1], self.input_size[2]], name="image_inputs")

        # Todo
        # pretrain model 추가
        self.image_outputs = tf.layers.conv2d(self.image_inputs, filters=[3, 3], kernel_size=[1, 1], strides=[1, 1], activation='relu')

    def _build_language_network(self):
        self.question_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.max_len], name="question_inputs")

        char_embedding = tf.get_variable("char_embedding", [self.char_size, self.embedding_size])
        embedding = tf.nn.embedding_lookup(char_embedding, self.question_inputs)

        with tf.variable_scope("language_model"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

            (fw_outputs, bw_outputs), states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedding, dtype=tf.float32)
            concated_outputs = tf.concat([fw_outputs, bw_outputs], axis=2)
            flatten = tf.reshape(concated_outputs, [-1, self.hidden_size * 2 * self.max_len])

        with tf.variable_scope("fully_connected_layer"):
            self.question_outputs = tf.layers.dense(inputs=flatten, units=196, activation=tf.nn.softmax)
