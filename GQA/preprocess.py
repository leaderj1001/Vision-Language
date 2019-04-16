import glob
import codecs
import json
import pprint

train_question_path = "D:/GQA/questions1.2/train/*.json"
test_question_path = "D:/GQA/questions1.2/test/*.json"


def get_filename():
    """
    Get filenames
    :return: train filenames, test filenames
    """
    train_filenames = glob.glob(train_question_path)
    test_filenames = glob.glob(test_question_path)

    return train_filenames, test_filenames


def load_data(train_filenames, test_filenames):

    train_question = {}
    for index, train_filename in enumerate(train_filenames):
        with codecs.open(train_filename, "r", encoding="utf-8", errors="ignore") as f:
            train_data = json.load(f)
            for line in train_data:
                train_question[line] = [train_data[line]["answer"], train_data[line]["imageId"], train_data[line]["question"]]

    test_question = {}
    for index, test_filename in enumerate(test_filenames):
        with codecs.open(test_filename, "r", encoding="utf-8", errors="ignore") as f:
            test_data = json.load(f)
            for line in test_data:
                test_question[line] = [test_data[line]["answer"], test_data[line]["imageId"], test_data[line]["question"]]

    return train_question, test_question


train_filenames, test_filenames = get_filename()

load_data(train_filenames, test_filenames)



