import argparse


def get_args():

    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--train-question-path", type=str, default="D:/Dataset/gqa/questions1.2/train_balanced/*.json", help="train question data path")
    parser.add_argument("--test-question-path", type=str, default="D:/Dataset/gqa/questions1.2/test-dev/*.json", help="test question data path")
    parser.add_argument("--image-path", type=str, default="D:/Dataset/gqa/images/images", help="train/test image data path")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min_word_length", type=int, default=1)
    parser.add_argument("--max_word_length", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=1852)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint_balanced')
    parser.add_argument("--checkpoint_name", type=str, default='vision_lang')
    # Todo
    # max_len 찾아야함.

    args = parser.parse_args()

    return args
