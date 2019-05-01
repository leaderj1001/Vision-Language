import tensorflow as tf
import numpy as np

import tensornets as nets
import tensorflow_hub as hub

QUESTION_LEN = 32
ANSWER_NUM = 1852

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

# reference,
# https://github.com/taki0112/Self-Attention-GAN-Tensorflow
def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def hw_flatten(x):
    return tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2], x.shape[-1]])


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=None)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                 kernel_initializer=weight_init,
                                 kernel_regularizer=None,
                                 strides=stride, use_bias=use_bias)

        return x


class Model(object):
    def __init__(self, sess, input_size, image_pretrained):
        self.sess = sess
        self.input_size = input_size
        self.image_pretrained = image_pretrained

        self.weights_init = tf.contrib.layers.xavier_initializer()

        self._build_image_network()
        self._build_language_network()
        self._build_model()

    def _build_image_network(self):
        self.image_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size[0], self.input_size[1], self.input_size[2]], name="image_input")
        if self.image_pretrained == "resnet":
            self.image_model = nets.resnets.resnet50(self.image_input)
        elif self.image_pretrained == "densenet":
            self.image_model = nets.densenets.densenet121(self.image_input)

        self.image_output = self.image_model.get_middles()[-1]
        # for img_output in self.image_outputs:
        #     print(img_output.shape)

        # self.attention_output = self.attention(self.image_output, ch=1024)

    def _load_model(self):
        self.sess.run(self.image_model.pretrained())

    def _build_language_network(self):
        self.question_input = tf.placeholder(dtype=tf.string, shape=[None, ], name="question_input")

        self.question_output = elmo(self.question_input, signature="default", as_dict=True)["elmo"]

        # char_embedding = tf.get_variable("char_embedding", [self.voca_size, self.embedding_size], initializer=tf.constant_initializer(pretrained_embedding))
        # embedding = tf.nn.embedding_lookup(char_embedding, self.question_input)
        #
        # with tf.variable_scope("language_model"):
        #     fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        #     bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        #
        #     (fw_outputs, bw_outputs), states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedding, dtype=tf.float32)
        #     concated_outputs = tf.concat([fw_outputs, bw_outputs], axis=2)
        #     flatten = tf.reshape(concated_outputs, [-1, self.hidden_size * 2 * self.max_len])
        #
        # with tf.variable_scope("fully_connected_layer"):
        #     self.question_outputs = tf.layers.dense(inputs=flatten, units=196, activation=tf.nn.softmax)

    def attention(self, x, ch, sn=False, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
            g = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
            h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=[-1, x.shape[1], x.shape[2], x.shape[3]])  # [bs, h, w, C]
            x = gamma * o + x

        return x

    def _build_model(self):
        # question feature
        question_feature = self.question_output
        question_feature = tf.expand_dims(question_feature, 2)
        question_feature = tf.expand_dims(question_feature, 2)

        # image feature
        image_feature = self.image_output
        image_feature = tf.expand_dims(image_feature, 1)

        # question image feature aggregation
        token_weight = tf.get_variable('token_weight', shape=[1, QUESTION_LEN, 7, 7, 1], dtype=tf.float32, trainable=True)
        question_image_feature = tf.math.multiply(question_feature, image_feature)
        question_image_feature = tf.math.multiply(question_image_feature, token_weight)
        question_image_feature = tf.reduce_mean(question_image_feature, axis=1)

        # attention
        question_image_feature = self.attention(question_image_feature, ch=1024)
        question_image_feature = tf.reduce_mean(question_image_feature, axis=[1, 2])

        self.answer_vec = tf.layers.dense(question_image_feature, ANSWER_NUM)
