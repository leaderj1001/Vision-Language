import tensorflow as tf
import numpy as np
import os
import time

from model import Model
from preprocess import load_all_question, batch_iterator
from config import get_args

# Image
# img = model.image_model.preprocess(image_test_x)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    with tf.Session() as sess:
        _, test_questions = load_all_question()
        print(len(test_questions))

        model = Model(sess, [224, 224, 3], "densenet", len(train_questions))
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))

        hit = 0
        total = 0
        for batch_image, batch_question, batch_question_length, batch_answer in batch_iterator(test_questions, args.batch_size):
            test_accuracy = sess.run(model.accuracy, feed_dict={
                model.image_input: batch_image,
                model.question_input: batch_question,
                model.question_length: batch_question_length,
                model.target: batch_answer
            })
            hit += test_accuracy * batch_image.shape[0]
            total += batch_image.shape[0]
            print(test_accuracy)
        print('test accuracy: ', hit / total)


if __name__ == "__main__":
    args = get_args()
    main(args)
