import glob
import codecs
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import mmh3
import re
import random
import pandas as pd
import os

import config

ANSWER_NUM = 1852


def load_answer_dict():
    with open("./answer_dict.json", "r") as f:
        load_dict = json.load(f)

    return load_dict


def convert_integer(data, dictionary):
    data = pd.DataFrame(data)

    data.replace(dictionary, inplace=True)

    data = list(np.squeeze(np.array(data)))
    one_hot_data = np.eye(ANSWER_NUM)[data]

    return one_hot_data


answer_dict = load_answer_dict()
args = config.get_args()
re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')


def get_filename():
    """
    Get filenames
    :return: train filenames, test filenames
    """
    train_filenames = glob.glob(args.train_question_path)
    test_filenames = glob.glob(args.test_question_path)

    return train_filenames, test_filenames


def get_image_filename():
    image_data_dict = {}
    for path in tqdm(glob.glob(args.image_path), mininterval=1, desc="make image_data_dict"):
        image_data_dict[path.split('/')[2].split('\\')[1].split('.')[0]] = path
    return image_data_dict


def read_image(path):
    image = np.array(Image.open(path))
    # shape, (224, 224, 3)
    resized_image = cv2.resize(image, dsize=(args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
    return resized_image


# def sentence_to_vec(sentence):
#     sub_sentence = re_sc.sub(' ', sentence).strip().split()
#
#     words = [w.strip() for w in sub_sentence]
#
#     words = [w for w in words if len(w) >= args.min_word_length and len(w) <= args.max_word_length]
#     if not words:
#         return [None] * 2
#     hash_func = lambda x: mmh3.hash(x, seed=17)
#     x = [hash_func(w) % 100001 for w in words]
#     temp = np.zeros(args.max_len, dtype=np.float32)
#
#     for i in range(len(x)):
#         temp[i] = x[i]
#     return temp


def sentence_padding(sentence):
    sub_sentence = re_sc.sub(' ', sentence).strip().split()

    words = [w.strip() for w in sub_sentence]

    words = [w for w in words if len(w) >= args.min_word_length and len(w) <= args.max_word_length]
    init_length = len(words)

    for _ in range(init_length, args.max_len):
        words.append("")
    return words, init_length


def load_question(filename):
    questions = {}
    with codecs.open(filename, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
        for line in tqdm(data):
            question = data[line]
            questions[line] = [question['answer'], question['imageId'], question['question']]

    return questions


def load_all_question():
    train_filenames, test_filenames = get_filename()

    train_question = {}
    for index, train_filename in tqdm(enumerate(train_filenames), mininterval=1, desc="load train question"):
        train_question.update(load_question(train_filename))

    test_question = {}
    for index, test_filename in tqdm(enumerate(test_filenames), mininterval=1, desc="load test question"):
        test_question.update(load_question(test_filename))

    return train_question, test_question


# def load_data():
#     train = []
#     test = []
#
#     image_data_dict = get_image_filename()
#     train_question, test_question = load_question()
#
#     for tq in tqdm(train_question, mininterval=1, desc="load train data (image, question, ground truth)"):
#         image = read_image(image_data_dict[tq[1]])
#         train.append([image, sentence_to_vec(tq[2]), tq[0]])
#
#     for teq in tqdm(test_question, mininterval=1, desc="load test data (image, question, ground truth)"):
#         image = read_image(image_data_dict[teq[1]])
#         train.append([image, sentence_to_vec(teq[2]), teq[0]])
#
#     shuffled_train = np.random.shuffle(train)
#     shuffled_test = np.random.shuffle(test)
#
#     train_x = shuffled_train[:2]
#     train_y = shuffled_train[2]
#
#     test_x = shuffled_test[:2]
#     test_y = shuffled_test[2]
#
#     return train_x, train_y, test_x, test_y

def batch_iterator(question_dict, batch_size, shape=(224, 224)):
    keys = list(question_dict.keys())
    random.shuffle(keys)
    while len(keys) != 0:
        batch_keys = keys[:batch_size]

        images = []
        questions = []
        questions_length = []
        answers = []

        for key in batch_keys:
            q_dict = question_dict[key]

            # image = np.array(Image.open(args.image_path + '/' + q_dict[1] + '.jpg').resize(shape))
            image = cv2.imread(args.image_path + '/' + q_dict[1] + '.jpg')
            image = cv2.resize(image, dsize=shape)
            question, question_length = sentence_padding(q_dict[2])
            answer = q_dict[0]

            images.append(image)
            questions.append(question)
            questions_length.append(question_length)
            answers.append(answer)

        images = np.array(images)
        questions = np.array(questions)
        questions_length = np.array(questions_length)
        one_hot_answers = convert_integer(answers, answer_dict)
        yield images, questions, questions_length, one_hot_answers

        del keys[:batch_size]


# question = load_question("D:/Dataset/gqa/questions1.2/train_all_questions/train_all_questions_0.json")
# question = load_question("D:/GQA/questions1.2/train/train_all_questions_0.json")
# batch = 64
#
# for i, q, q_l, a in batch_iterator(question, batch):
#     print(i.shape)
#     print(q.shape)
#     print(q_l.shape)
#     print(a)
