import os

from brain_tumor_training_multi_feature import train_run as multi_train_run


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    multi_train_run()
