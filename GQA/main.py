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
#     os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        if not os.path.isdir('reporting'):
            os.mkdir('reporting')
        writer = tf.summary.FileWriter('./reporting', sess.graph)

        model = Model(sess, [224, 224, 3], image_pretrained="densenet")
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        try:
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
            print('checkpoint restored. train from checkpoint')
        except:
            print('failed to load checkpoint. train from the beginning')

        train_questions, test_questions = load_all_question()
        # start_time = time.time()
        for epoch in range(1, args.epochs):
            for batch_image, batch_question, batch_question_length, batch_answer in batch_iterator(train_questions, args.batch_size):
                gstep, _, train_loss, train_accuracy, t_l_summary, t_a_summary = sess.run([model.global_step, model.train_op, model.loss, model.accuracy, model.train_loss_summary, model.train_acc_summary], feed_dict={
                    model.image_input: batch_image,
                    model.question_input: batch_question,
                    model.question_length: batch_question_length,
                    model.target: batch_answer
                })
                # hit = np.array(np.argmax(y_pred, axis=1) == np.argmax(batch_answer, axis=1))
                # train_acc = np.sum(hit) / hit.shape[0]
                print('gstep: ' + str(gstep) + ' epoch ' + str(epoch) + ' train_acc: ' + str(train_accuracy) + ' loss: ' + str(train_loss))

                writer.add_summary(t_l_summary, gstep)
                writer.add_summary(t_a_summary, gstep)
                if gstep % 1000 == 0:
                    print('save to checkpoint')
                    saver.save(sess, save_path=args.checkpoint_dir + '/' + args.checkpoint_name, global_step=model.global_step)

            # training time
            # time_interval = time.time() - start_time
            # time_split = time.gmtime(time_interval)
            # print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min,
            #       "Second: ", time_split.tm_sec)

            # for batch_image, batch_question, batch_question_length, batch_answer in batch_iterator(test_questions, args.batch_size):
            #     gstep, _, test_loss, test_accuracy, test_l_summary, test_a_summary = sess.run([model.global_step, model.train_op, model.loss, model.accuracy, model.test_loss_summary, model.test_acc_summary], feed_dict={
            #         model.image_input: batch_image,
            #         model.question_input: batch_question,
            #         model.question_length: batch_question_length,
            #         model.target: batch_answer
            #     })
            #     # test_hit = np.array(np.argmax(test_y_pred, axis=1) == np.argmax(batch_answer, axis=1))
            #     # test_acc = np.sum(test_hit) / hit.shape[0]
            #     print('validation ' + str(epoch) + ' test_acc: ' + str(test_accuracy) + ' loss: ' + str(test_loss))
            #
            #     writer.add_summary(test_l_summary, gstep)
            #     writer.add_summary(test_a_summary, gstep)


if __name__ == "__main__":
    args = get_args()
    main(args)
