import os
import json

import requests
import numpy as np
import pandas as pd

from preprocessor.preprocessing import *
from utils.api_authen import load_academic_research_bearer
from utils.utils import get_params, timing, chunks
from utils.logger import logger

cwd = os.getcwd()  # current working directory
config_dir = os.path.join(cwd, 'crawler', 'config', 'fields_expansions')

api_name = "migrationsKB"
bearer_token = load_academic_research_bearer(cwd, api_name)


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def create_url(ids, tweet_fields):
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    ids = "ids=" + ",".join(ids)
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def main(ids, save_path):
    tweets_fields = "tweet.fields=lang,author_id,text,entities"
    # ids = ["1026518840195383298", "1060146932910305280"]
    url = create_url(ids, tweets_fields)
    json_response = connect_to_endpoint(url)
    with open(save_path, "w") as writer:
        json.dump(json_response, writer)


def preprocess_pl(label_dict, data_dir, save_dir=None, dataset=None):
    df_ls = []
    for file in os.listdir(data_dir):
        # if dataset in file:
        filepath = os.path.join(data_dir, file)
        with open(filepath) as f:
            json_data = json.load(f)
            if "data" in json_data:
                df_ = pd.DataFrame.from_dict(json_data['data'])
                df_ls.append(df_)
    df = pd.concat(df_ls)
    print(df)
    for id_, label in label_dict.items():
        df.loc[df['id'] == id_, "label"] = label
    df["text"] = df["text"].apply(preprocessing_one_tweet)
    df = df[['id', 'text', 'label']]
    print(df.label.value_counts())
    # test - 0:726, 1:10, 2: 62
    # training - 0:2184, 1:32, 2:78
    # df.to_csv(os.path.join(save_dir, f"{dataset}.csv"), index=False)
    return df
    # # post.
    # data_dir = "datasets/hate_speech_detection/cyberbullying-Polish"
    # training_file = os.path.join(data_dir, "preprocessed", "training.csv")
    # test_file = os.path.join(data_dir, "preprocessed", "test.csv")
    # train = pd.read_csv(training_file)
    # test = pd.read_csv(test_file)
    # df = pd.concat([train, test])
    # print(df.label.value_counts())
    # df = df.groupby("label").sample(167)
    # df = shuffle(df)
    # df.to_csv(os.path.join(data_dir, "preprocessed", "balanced_all.csv"))


def crawl_and_preprocess_pl(crawl=True):
    data_dir = "datasets/hate_speech_detection/cyberbullying-Polish"
    to_crawl_dir = "datasets/hate_speech_detection/cyberbullying-Polish/task02"
    # to_crawl_file = os.path.join(to_crawl_dir)
    dataset = "training"
    filename = f"{dataset}_set_clean_only_ids.txt"
    with open(os.path.join(to_crawl_dir, filename)) as f:
        tweet_ids = [x.replace('\n', '') for x in f.readlines()]
    # print(tweet_ids)
    if crawl:
        chunks_ = chunks(tweet_ids, n=10)
        for id_, i in enumerate(list(chunks_)[897:]):
            save_path = os.path.join("datasets/hate_speech_detection/cyberbullying-Polish/crawled",
                                     f"{dataset}_set_{id_ + 897}.json")
            main(i, save_path)
    else:
        with open(os.path.join(to_crawl_dir, f"{dataset}_set_clean_only_tags.txt")) as f:
            labels = [x.replace('\n', '') for x in f.readlines()]
        id_label_dict = dict(zip(tweet_ids, labels))
        preprocess_pl(id_label_dict, os.path.join(data_dir, "crawled"), os.path.join(data_dir, "preprocessed"), dataset)


def crawl_and_preprocess_it(crawl=False, pre=True, post=True):
    data_dir = "datasets/hate_speech_detection/it"
    df = pd.read_csv(os.path.join(data_dir, "IHSC_ids.tsv"), sep="\t")

    if crawl:
        tweet_ids = [str(x) for x in df['tweet_id']]
        chunks_ = chunks(tweet_ids, n=20)
        for id_, i in enumerate(list(chunks_)[299:]):
            save_path = os.path.join(data_dir, "crawled",
                                     f"{id_ + 299}.json")
            main(i, save_path)
    if pre:
        emotions = ["hs", "aggressiveness", "offensiveness", "irony", "stereotype"]
        df.loc[df['hs'] == "yes", "label"] = 2  # hatespeech
        df.loc[df['offensiveness'] == "strong", "label"] = 1  # offensive
        df.loc[(df['hs'] == "no") & (df['offensiveness'] == "no") & (df['irony'] == "no") & (df['stereotype'] == "no"),
               "label"] = 0  # normal
        df.dropna(axis=0, subset=["label"], inplace=True)
        label_dict = dict(zip([str(x) for x in df['tweet_id']], df['label']))
        print(label_dict)
        data = preprocess_pl(label_dict, os.path.join(data_dir, "crawled"))
        data = data.dropna(subset=["label"])
        data.to_csv(os.path.join(data_dir, "preprocessed", "all.csv"))

        # balanced
        df = df.groupby("label").sample(246)  # nr. of offensive tweets.
        df = shuffle(df)
        df.to_csv(os.path.join(data_dir, "preprocessed", "balanced_all.csv"))


if __name__ == '__main__':
    crawl_and_preprocess_it(crawl=False, pre=True)
