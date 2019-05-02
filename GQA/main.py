import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import Model
from preprocess import get_filename, load_all_question, batch_iterator
from config import get_args


def main():
    args = get_args()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model = Model(sess, [224, 224, 3], image_pretrained="densenet")
        sess.run(tf.global_variables_initializer())

        # Image
        # img = model.image_model.preprocess(image_test_x)

        for epoch in range(1, args.epochs):
            train_questions, test_questions = load_all_question()

            for batch_image, batch_question, batch_question_length, batch_answer in batch_iterator(train_questions, args.batch_size):
                _, y_pred, loss = sess.run([model.train, model.y_pred, model.loss], feed_dict={
                    model.image_input: batch_image,
                    model.question_input: batch_question,
                    model.question_length: batch_question_length,
                    model.target: batch_answer
                })
                hit = np.array(np.argmax(y_pred, axis=1) == np.argmax(batch_answer, axis=1))
                train_acc = np.sum(hit) / hit.shape[0]
                print('epoch ' + str(epoch) + ' train_acc: ' + str(train_acc) + ' loss: ' + str(loss))
            saver.save('./checkpoint/vision_lang', global_step=model.global_step)
            # for test_filename in test_filenames:
            #     test_question_json = load_question(test_filename)
            #     for batch_image, batch_question, batch_question_length, answer in tqdm(batch_iterator(test_question_json, args.batch_size)):
            #         pass


if __name__ == "__main__":
    main()
