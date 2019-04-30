import glob
import codecs
import json
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import cv2
import pprint
import mmh3
import re
from collections import Counter

import config

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


def sentence_to_vec(sentence):
    sub_sentence = re_sc.sub(' ', sentence).strip().split()

    words = [w.strip() for w in sub_sentence]

    words = [w for w in words if len(w) >= args.min_word_length and len(w) <= args.max_word_length]
    if not words:
        return [None] * 2
    hash_func = lambda x: mmh3.hash(x, seed=17)
    x = [hash_func(w) % 100001 for w in words]
    temp = np.zeros(args.max_len, dtype=np.float32)

    for i in range(len(x)):
        temp[i] = x[i]
    return temp


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
        train.append([image, sentence_to_vec(tq[2]), tq[0]])

    for teq in tqdm(test_question, mininterval=1, desc="load test data (image, question, ground truth)"):
        image = read_image(image_data_dict[teq[1]])
        train.append([image, sentence_to_vec(teq[2]), teq[0]])

    shuffled_train = np.random.shuffle(train)
    shuffled_test = np.random.shuffle(test)

    train_x = shuffled_train[:2]
    train_y = shuffled_train[2]

    test_x = shuffled_test[:2]
    test_y = shuffled_test[2]

    return train_x, train_y, test_x, test_y
