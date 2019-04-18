import glob
import codecs
import json
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import cv2
import pprint

import config

args = config.get_args()


# def decompose_as_one_hot(in_char, warning=True):
#     """
#     If in_char is in ASCII code, it is converted to ASCII code number.
#     If it is not, convert it to a fixed number.
#     :param in_char: Each char, example) 'Hello', 'H', 'e', 'l', 'l', 'o'
#     :return: number
#     """
#     if in_char < 128:
#         result = in_char  # 67~
#     else:
#         if warning:
#             print('Unhandled character:', chr(in_char), in_char)
#         # unknown character
#         result = 128 + 1  # for unknown character
#
#     return [result]
#
#
# def decompose_string_to_one_hot(string, warning=True):
#     """
#     Using decompose_as_one_hot function, convert one_hot mapping
#     :param string: Each input sentence data
#     :return: number vector
#     """
#     one_hot_string = []
#     for char in string:
#         decomposed_char = decompose_as_one_hot(ord(char), warning=warning)
#         one_hot_string.extend(decomposed_char)
#
#     return one_hot_string
#
#
# def data_to_vector(train, test, max_length):
#     """
#     If a string is less than maxlen fill it with 0, otherwise, use char to maxlen.
#     :param max_length:
#     :return:
#     """
#     one_hot_train = np.zeros((len(train), max_length), dtype=np.int32)
#     one_hot_test = np.zeros((len(test), max_length), dtype=np.int32)
#
#     for idx, data in enumerate(train):
#         length = len(data)
#         sentence_vector = decompose_string_to_one_hot(data)
#         if length >= max_length:
#             length = max_length
#             one_hot_train[idx, :length] = np.array(sentence_vector)[:length]
#         else:
#             one_hot_train[idx, :length] = np.array(sentence_vector)
#
#     for idx, data in enumerate(test):
#         length = len(data)
#         sentence_vector = decompose_string_to_one_hot(data)
#         if length >= max_length:
#             length = max_length
#             one_hot_test[idx, :length] = np.array(sentence_vector)[:length]
#         else:
#             one_hot_test[idx, :length] = np.array(sentence_vector)
#
#     return one_hot_train, one_hot_test


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


def load_question():
    train_filenames, test_filenames = get_filename()

    train_question = {}
    for index, train_filename in tqdm(enumerate(train_filenames), mininterval=1, desc="load train question"):
        with codecs.open(train_filename, "r", encoding="utf-8", errors="ignore") as f:
            train_data = json.load(f)
            for line in train_data:
                train_question[line] = [train_data[line]["answer"], train_data[line]["imageId"], train_data[line]["question"]]

    test_question = {}
    for index, test_filename in tqdm(enumerate(test_filenames), mininterval=1, desc="load test question"):
        with codecs.open(test_filename, "r", encoding="utf-8", errors="ignore") as f:
            test_data = json.load(f)
            for line in test_data:
                test_question[line] = [test_data[line]["answer"], test_data[line]["imageId"], test_data[line]["question"]]

    return train_question, test_question


def load_data():
    train = []
    test = []

    image_data_dict = get_image_filename()
    train_question, test_question = load_question()

    for tq in tqdm(train_question, mininterval=1, desc="load train data (image, question, ground truth)"):
        image = read_image(image_data_dict[tq[1]])
        train.append([image, tq[2], tq[0]])

    for teq in tqdm(test_question, mininterval=1, desc="load test data (image, question, ground truth)"):
        image = read_image(image_data_dict[teq[1]])
        train.append([image, teq[2], teq[0]])

    shuffled_train = np.random.shuffle(train)
    shuffled_test = np.random.shuffle(test)

    train_x = shuffled_train[:2]
    train_y = shuffled_train[2]

    test_x = shuffled_test[:2]
    test_y = shuffled_test[2]

    return train_x, train_y, test_x, test_y
