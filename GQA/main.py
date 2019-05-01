import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import Model
from preprocess import get_filename, load_all_question, batch_iterator
from config import get_args


def main():
    args = get_args()

    with tf.Session() as sess:
        model = Model(sess, [224, 224, 3], image_pretrained="densenet")
        sess.run(tf.global_variables_initializer())

        # Image
        # img = model.image_model.preprocess(image_test_x)

        for epoch in range(1, args.epochs):
            train_questions, test_questions = load_all_question()

            train_correct = 0
            for batch_image, batch_question, batch_question_length, batch_answer in tqdm(batch_iterator(train_questions, args.batch_size)):
                _, y_pred, loss = sess.run([model.train, model.y_pred, model.loss], feed_dict={
                    model.image_input: batch_image,
                    model.question_input: batch_question,
                    model.question_length: batch_question_length,
                    model.target: batch_answer
                })

                print(train_correct, loss)
                train_correct += (np.argmax(y_pred) == np.argmax(batch_answer)).sum()
                print(train_correct, loss)

            # for test_filename in test_filenames:
            #     test_question_json = load_question(test_filename)
            #     for batch_image, batch_question, batch_question_length, answer in tqdm(batch_iterator(test_question_json, args.batch_size)):
            #         pass


if __name__ == "__main__":
    main()
