import tensorflow as tf
import numpy as np

import tensornets as nets

from model import Model
from preprocess import load_data
from config import get_args


def train(train_x, train_y, args):
    total_index = len(train_x) // args.batch_size

    for batch_index in range(total_index):
        batch_train_x, batch_train_y = train_x[batch_index: ]


def main():
    args = get_args()
    train_x, train_y, test_x, test_y = load_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model = Model(sess, [224, 224, 3], max_len=16, voca_size=16, embedding_size=32)

        for epoch in range(1, args.epochs):
            pred = sess.run(model.image_model, feed_dict={model.image_inputs: model.image_model.preprocess(img)})
            print(pred.shape)


if __name__ == "__main__":
    main()