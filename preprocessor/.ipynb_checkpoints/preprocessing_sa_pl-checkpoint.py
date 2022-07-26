import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

config = yaml.load(open(os.path.join('datasets', 'config.yaml')), yaml.FullLoader)
TRAIN_LEN = config["xlm-t"]["train_len"]
VAL_LEN = config["xlm-t"]["val_len"]
TEST_LEN = config["xlm-t"]["test_len"]
ALL_LEN = config['xlm-t']['LEN']
print(ALL_LEN)


def read_data():
    data_dir = "datasets/sentiment_analysis/pl/"
    save_dir = os.path.join(data_dir, "preprocessed")
    filename = "polish_sentiment_dataset.csv"

    df = pd.read_csv(os.path.join(data_dir, filename))
    df.rename(columns={'rate': 'label', "description": "text"}, inplace=True)

    df['label'].replace(to_replace=[-1, 0, 1], value=[0, 1, 2], inplace=True)
    df['text'] = df['text'].str.replace("\n", "")
    print(df.label.value_counts())
    print(df[df.label == 1])
    df = df[df['length'] > 5]
    print(df.label.value_counts())

    print(df.columns)


def read_sentences_files(dataset):
    """
    * minus_m : strong negative
    * zero --neutral
    * plus_m :strong positive.
    :param dataset:
    :return:
    """
    data_dir = "datasets/sentiment_analysis/pl/"
    save_dir = os.path.join(data_dir, "preprocessed")
    filepath = os.path.join(data_dir, "dataset", f"all2.sentence.{dataset}.txt")
    texts = []
    labels = []
    with open(filepath) as f:
        for line in f.readlines():
            text, label_ = line.replace('\n', '').split('__label__')
            if label_ == "z_minus_m":
                label = 0
            if label_ == "z_zero":
                label = 1
            if label_ == "z_plus_m":
                label = 2
            texts.append(text)
            labels.append(label)
    df = pd.DataFrame.from_dict({"text": texts, "label": labels})
    return df


def data_sample(df, LEN):
    df_neutral = df[df['label'] == 1].sample(LEN, random_state=1)
    df_positive = df[df['label'] == 2].sample(LEN, random_state=1)
    df_negative = df[df['label'] == 0].sample(LEN, random_state=1)
    df_ = pd.concat([df_neutral, df_negative, df_positive])
    return shuffle(df_)


if __name__ == '__main__':
    dataset = "test"
    data_dir = "datasets/sentiment_analysis/pl/"
    save_dir = os.path.join(data_dir, "preprocessed")
    df = read_sentences_files(dataset)
    print(df.label.value_counts())
    print(VAL_LEN)
    df = data_sample(df, int(TEST_LEN/3))
    print(df.label.value_counts())
    df.to_csv(os.path.join(save_dir, f"{dataset}.csv"), index=False)

