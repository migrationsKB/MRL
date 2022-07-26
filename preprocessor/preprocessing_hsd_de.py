import os
import pandas as pd
from preprocessor.preprocessing import *


def read_data(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    df = pd.read_csv(data_path, sep="\t")
    print(df.task_1.value_counts())
    print(df.task_2.value_counts())

    df.loc[df['task_2'] == "NONE", "label"] = 0  # normal
    df.loc[df['task_2'] == "OFFN", "label"] = 1  # offensive
    df.loc[df['task_2'] == "HATE", "label"] = 2  # hatespeech
    df.text = df.text.apply(preprocessing_one_tweet)

    return df[['text', 'label']]


def read_data_IWG():
    datapath = "datasets/hate_speech_detection/IWG_de/germanHatespeechRefugees.csv"
    df = pd.read_csv(datapath).rename(columns={"HatespeechOrNot (Expert 1)": "Anno1",
                                               "HatespeechOrNot (Expert 2)": "Anno2",
                                               "Hatespeech Rating (Expert 2)": "Rating"})
    print(df.iloc[0]["Anno1"] == "YES")
    df['Rating'] = df['Rating'].astype(int)
    print(df.dtypes)
    df.loc[(df['Anno1'].str.contains("NO") & df['Anno2'].str.contains("NO") & df['Rating']==1), "label"] = 0
    # df.loc[( df['Rating'] == 1), "label"] = 0

    print("normal ratings:", df[df['label']==0]['Rating'].value_counts())
    print("*"*50)

    # df.loc[(df['Anno1'].str.contains("YES") & df['Anno2'].str.contains("YES")), "label"] = 1
    # df.loc[(
    #                (df['Anno1'].str.contains("YES") & df['Anno2'].str.contains("NO")) |
    #                (df['Anno1'].str.contains("NO") & df['Anno2'].str.contains("YES"))
    #        )
    # , "label"] = "unknown"
    df.loc[(df['Rating']>=5), "label"]=2
    df['label'] = df['label'].fillna(1)
    print(df.label.value_counts())

    #
    # df_unknown = df[df['label']=="unknown"]
    print(df.head())
    print(len(df))


def read_data_hasoc():
    data_dir = "datasets/hate_speech_detection/german_hasoc"
    save_dir = os.path.join(data_dir, "preprocessed")
    df_test = read_data(data_dir, "hasoc_de_test_gold.tsv")
    df_train = read_data(data_dir, "german_dataset.tsv")

    df = pd.concat([df_test, df_train])
    print(df.label.value_counts())
    df.drop_duplicates(subset='text', inplace=True)

    df = shuffle(df)
    df.to_csv(os.path.join(save_dir, "all.csv"))
    df_balanced = df.groupby('label').sample(152)
    df_balanced.to_csv(os.path.join(save_dir, "test_balanced.csv"))

    print(df.label.value_counts())


if __name__ == '__main__':
    # read_data_IWG()
    read_data_hasoc()

# 0.0    3982
# 1.0     286
# 2.0     152
# Name: label, dtype: int64
