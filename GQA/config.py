import argparse


def get_args():

    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--train-question-path", type=str, default="D:/GQA/questions1.2/train/*.json", help="train question data path")
    parser.add_argument("--test-question-path", type=str, default="D:/GQA/questions1.2/test/*.json", help="test question data path")
    parser.add_argument("--image-path", type=str, default="D:/GQA/images/*.jpg", help="train/test image data path")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min_word_length", type=int, default=1)
    parser.add_argument("--max_word_length", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    # Todo
    # max_len 찾아야함.

    args = parser.parse_args()

    return args