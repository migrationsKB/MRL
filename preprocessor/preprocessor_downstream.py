##########################################
## preprocess data from datasets #########
##########################################
import os
import argparse

import polars as pl
from sklearn.utils import shuffle

from preprocessor.preprocessing import *

parser = argparse.ArgumentParser(description="Preprocessing datasets for downstream tasks")
parser.add_argument("--dataset", default="xlm-t", help="dataset name")
parser.add_argument("--task", default="SA", help="either SA (Sentiment analysis) or HSD (hate speech detection)")
parser.add_argument("--savepath", default="xlm-t", help="where to save the preprocessed files.")
args = parser.parse_args()

datasets_config = {
    "GermEval2017": "datasets/sentiment_analysis/GermEval2017",
    "SemEval2017": "datasets/sentiment_analysis/SemEval2017/datastories-semeval2017-task4-master/dataset/Subtask_A/downloaded",
    "xlm-t": "datasets/sentiment_analysis/xlm-t/data/sentiment"
}


def process_one_file(filepath, dataset, sep="\t", has_header=False):
    df = pl.read_csv(filepath, sep=sep, has_header=has_header)
    if dataset == "SemEval2017":
        df.columns = ['tweet_id', 'sentiment', 'text']  # semeval2017
    if dataset == "GermEval2017":
        df.columns = ['tweet_id', 'text', 'Relevance', 'sentiment', 'aspect:polarity']  # GermEval2017
        # df = df[df['tweet_id'].str.contains('twitter')]  # if the id contains "twitter"

    df['preprocessed_text'] = df['text'].apply(preprocessing_one_tweet)
    df = df.drop_nulls(subset='preprocessed_text')
    df = df.drop_duplicates(subset='preprocessed_text')
    df = df[['sentiment', 'preprocessed_text']]
    return df


def get_balanced_dataset(df):
    """
    According to the lowest number of samples of the class, to sample other classes.
    :param df:
    :return:
    """
    df = shuffle(df)

    len_group = df.groupby('sentiment')['preprocessed_text'].count()['preprocessed_text_count']
    min_group = min(len_group)
    print(min_group)
    df_positive = df[df['sentiment'] == 'positive'].sample(n=min_group)
    df_neutral = df[df['sentiment'] == 'neutral'].sample(n=min_group)
    df_negative = df[df['sentiment'] == 'negative'].sample(n=min_group)
    new_df = pl.concat([df_positive, df_negative, df_neutral])
    new_df = shuffle(new_df)
    assert len(new_df) == min_group * 3
    return new_df


def preprocess_semeval2017():
    dataset_path = datasets_config["SemEval2017"]
    save_dir = os.path.join('datasets/sentiment_analysis/SemEval2017/', 'preprocessed')
    filepath = "datasets/sentiment_analysis/SemEval2017/datastories-semeval2017-task4-master/dataset/Subtask_A/4A-English/SemEval2017-task4-dev.subtask-A.english.INPUT.txt"
    # filepath = "datasets/sentiment_analysis/SemEval2017/datastories-semeval2017-task4-master/dataset/Subtask_A/gold/SemEval2017-task4-test.subtask-A.english.txt"
    dataset_type = "dev"
    df_ls = []
    df = process_one_file(filepath, "SemEval2017")
    print(filepath)
    df_ls.append(df)
    # for file in os.listdir(dataset_path):
    #     if file.endswith('.tsv'):
    #         # if dataset_type in file:
    #             filepath = os.path.join(dataset_path, file)
    #             print(filepath)
    #             df = process_one_file(filepath, dataset="SemEval2017")
    #             df_ls.append(df)

    if len(df_ls) > 0:
        concatDf = pl.concat(df_ls)
        concatDf = concatDf.drop_duplicates(subset='preprocessed_text')
        print(len(concatDf))
        concatDf.to_csv(os.path.join(save_dir, f'{dataset_type}.csv'), sep=',')

        balanced_df = get_balanced_dataset(concatDf)
        print(len(balanced_df))
        balanced_df.to_csv(os.path.join(save_dir, f'{dataset_type}_balanced.csv'), sep=',')


def preprocess_germeval2017():
    dataset_path = datasets_config[args.dataset]
    save_dir = os.path.join('datasets/sentiment_analysis/GermEval2017/', 'preprocessed')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    dataset_type = "dev"
    df_ls = []
    for file in os.listdir(dataset_path):
        if file.endswith('.tsv'):
            if dataset_type in file:
                filepath = os.path.join(dataset_path, file)
                print(filepath)
                df = process_one_file(filepath, dataset="GermEval2017")
                df_ls.append(df)

    if len(df_ls) > 0:
        concatDf = pl.concat(df_ls)
        concatDf = concatDf.drop_duplicates(subset='preprocessed_text')
        print(len(concatDf))
        concatDf.to_csv(os.path.join(save_dir, f'{dataset_type}.csv'), sep=',')

        balanced_df = get_balanced_dataset(concatDf)
        print(len(balanced_df))
        balanced_df.to_csv(os.path.join(save_dir, f'{dataset_type}_balanced.csv'), sep=',')


def preprocess_xlmt(dataset, lang):
    """
    Already balanced.
    :return:
    """
    dataset_path = datasets_config[args.dataset]
    save_dir = os.path.join('datasets/sentiment_analysis/xlm-t/', 'preprocessed')
    # dataset = 'test'
    text_dict = {}
    # for dirname in os.listdir(dataset_path):
    dirname = lang
    with open(os.path.join(dataset_path, dirname, f'{dataset}_text.txt')) as f:
        text_dict['text'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(dataset_path, dirname, f'{dataset}_labels.txt')) as f:
        text_dict['labels'] = [x.replace('\n', '') for x in f.readlines()]

    df = pd.DataFrame.from_dict(text_dict)

    # preprocess text
    df['text'] = df['text'].apply(preprocessing_one_tweet)
    print(df['text'])
    df = df.dropna()
    print(len(df))
    df = df.drop_duplicates(subset='text')
    print(len(df))

    save_dirn = os.path.join(save_dir, dirname)
    if not os.path.exists(save_dirn):
        os.mkdir(save_dirn)

    df.to_csv(os.path.join(save_dirn, f'{dataset}.csv'), index=False)


def preprocess_hu():
    dataset_path = "datasets/sentiment_analysis/hu/"
    save_dir = os.path.join(dataset_path, "preprocessed")
    datapath = os.path.join(dataset_path, "OpinHuBank_20130106.csv")
    df = pd.read_csv(datapath)
    print(df.head())



if __name__ == '__main__':
    pass
     # lang_code = "spanish"
    # preprocess_xlmt("train", lang_code)
    # preprocess_xlmt("test", lang_code)
    # preprocess_xlmt("val", lang_code)
