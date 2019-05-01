import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import glob

import tensornets as nets

from model import Model
from preprocess import load_data
from config import get_args


def train(train_x, train_y, args):
    total_index = len(train_x) // args.batch_size

    batch_train_x, batch_train_y = train_x[:16], train_y[:16]

    return batch_train_x, batch_train_y


def main():
    args = get_args()
    # image_test_x, sentence_test_x, test_y = load_data()
    # print(sentence_test_x.shape)

    # image_test_x = np.array(image_test_x, dtype=np.float32)
    # image_test_x = np.reshape(image_test_x, newshape=[-1, 224, 224, 3])

    # batch_train_x, batch_train_y = train(train_x, train_y, args)
    # print(batch_train_x.shape, batch_train_y.shape)

    sentence_test_x = ["Hello, I'm Myeongjun Kim. Nice to meet you.", ]
    image_test_x = np.zeros([1, 224, 224, 3], dtype=np.float32)

    with tf.Session() as sess:
        model = Model(sess, [224, 224, 3], image_pretrained="densenet")
        sess.run(tf.global_variables_initializer())

        # Image
        img = model.image_model.preprocess(image_test_x)
        attention_outputs = sess.run(model.attention_outputs, feed_dict={model.image_inputs: img})
        print(np.array(attention_outputs).shape)
        # question
        # question_out = sess.run(model.question_outputs, feed_dict={model.question_inputs: sentence_test_x})
        # print(np.array(middles[-1]).shape, np.array(question_out).shape)
        # print(np.array(question_out).shape)


if __name__ == "__main__":
    main()
