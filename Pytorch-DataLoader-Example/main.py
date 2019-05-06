import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import json
import cv2
import pickle
import pandas as pd

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def write_pickle(data, filename):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def sentence_padding(sentence):
    sub_sentence = sentence.strip().split()

    words = [w.strip() for w in sub_sentence]

    words = [w for w in words if len(w) >= 1 and len(w) <= 32]
    init_length = len(words)

    for _ in range(init_length, 32):
        words.append("")
    return words, init_length


def get_file_names():
    file_names = glob.glob("D:/GQA/questions1.2/train/*.json")
    return file_names


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def load_answer_dict():
    with open("./answer_dict.json", "r") as f:
        load_dict = json.load(f)

    return load_dict


def convert_integer(data, dictionary):
    data = pd.DataFrame(data)

    data.replace(dictionary, inplace=True)

    data = list(np.squeeze(np.array(data)))
    one_hot_data = np.eye(1852)[data]

    return one_hot_data


class DataSetLoader(Dataset):
    # initialize your own dataset.
    def __init__(self):
        self.data = load_pickle("./test.pkl")
        self.len = len(self.data)
        self.answer_dict = load_answer_dict()

    def __getitem__(self, idx):
        image = cv2.imread("D:/GQA/images/" + self.data[idx][1] + ".jpg", cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        split_sentence, sentence_length = sentence_padding(self.data[idx][2])
        one_hot_data = np.eye(1852)[self.answer_dict[self.data[idx][0]]]
        return image, split_sentence, sentence_length, one_hot_data

    def __len__(self):
        return self.len


def main():

    # make pickle data file
    # file_names = get_file_names()
    # file_names = [file_names[0]]
    #
    # pickle_list = []
    #
    # for file_name in file_names:
    #     print(file_name)
    #     data = load_json(file_name)
    #
    #     for line in data:
    #         # print(line, data[line]["answer"], data[line]["imageId"], data[line]["question"])
    #         pickle_list.append([data[line]["answer"], data[line]["imageId"], data[line]["question"]])
    # write_pickle(pickle_list, "test")

    dataset = DataSetLoader()
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=2)

    import time
    start_time = time.time()
    for image, question, question_length, target in train_loader:
        print(type(image), type(question), type(question_length), type(target))
        break
    print(time.time() - start_time)


if __name__ == "__main__":
    main()
