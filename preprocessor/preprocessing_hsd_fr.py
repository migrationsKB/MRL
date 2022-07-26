import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from preprocessor.preprocessing import *


def read_data(data_dir, filename, save_dir):
    df = pd.read_csv(os.path.join(data_dir, filename))
    print(df.sentiment.value_counts())
    df["text"] = df.tweet.apply(preprocessing_one_tweet)
    df["LEN"] = df["text"].str.len()
    df.drop_duplicates(subset='text', inplace=True)
    df = df[df['LEN'] > 10]
    print(df.sentiment.value_counts())

    print(df.iloc[0])

    df.loc[(df['sentiment'].str.contains("hateful") & ~df['sentiment'].str.contains("normal")), "label"] = 2
    df.loc[df['sentiment'] == "offensive", "label"] = 1
    df.loc[df['sentiment'] == "normal", "label"] = 0
    df.label.dropna(inplace=True)
    print(len(df))
    train_, test_dev_ = train_test_split(df, test_size=0.2)
    val_, test_ = train_test_split(test_dev_, test_size=0.5)
    train_.to_csv(os.path.join(data_dir, "train.csv"))
    val_.to_csv(os.path.join(data_dir, "val.csv"))
    test_.to_csv(os.path.join(data_dir, "test.csv"))

    len_normal = int(len(df[df['label'] == 2]) * 1.3)

    df_normal = df[df['label'] == 0].sample(len_normal, random_state=1)
    df_offensive = df[df['label'] == 1].sample(len_normal, random_state=1)
    df_hate = df[df['label'] == 2]

    df_ = pd.concat([df_normal, df_offensive, df_hate])
    df_ = shuffle(df_)
    df_ = df_[['text', 'label']]

    train, test_dev = train_test_split(df_, test_size=0.2)
    val, test = train_test_split(test_dev, test_size=0.5)

    print(train.label.value_counts())
    print(val.label.value_counts())
    print(test.label.value_counts())

    train.to_csv(os.path.join(save_dir, "fr/train.csv"))
    val.to_csv(os.path.join(save_dir, "fr/val.csv"))
    test.to_csv(os.path.join(save_dir, "fr/test.csv"))


if __name__ == '__main__':
    data_dir = "datasets/hate_speech_detection/hate_speech_mlma"
    save_dir = os.path.join(data_dir, "preprocessed")
    read_data(data_dir, "fr_dataset.csv", save_dir)
