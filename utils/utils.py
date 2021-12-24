import os
import time
import yaml
import json
from glob import glob

import pandas as pd
import numpy as np


def timing(fn):
    """
    Define a wrapper that can measure time for every function
    codes from @capeandcode
    :param fn:
    :return:
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args)
        end = time.time()
        print(f"The function {fn.__name__} took {end - start} to finish execution.")
        return ret

    return wrapper


def load_keywords_for_lang(root_dir: str, lang: str) -> list:
    """
    Load keywords for a specific language from config folder
    :param root_dir: given root directory
    :param lang: Language
    :return: a list of lower-cased deduplicated keywords.
    """
    filepath = os.path.join(root_dir, 'crawler', 'config', 'keywords', lang + '.csv')
    df = pd.read_csv(filepath)
    keywords = df['keyword'].str.lower()
    dedup = list(set(keywords))
    return dedup


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_keywords_by_category(category_dir='data/extracted/1st_round/hashtags_categories'):
    """
    Get the keywords by category
    :param category: category of the hashtags.
    :return:
    """
    keywords = []
    # go through hashtags categories load the keywords.
    for file in glob(category_dir + '/**.txt'):
        # print(file)
        with open(file) as reader:
            for line in reader.readlines():
                line = '#' + line.replace('\n', '')
                keywords.append(line)
    return list(set(keywords))


def get_params(cwd):
    """
    Loading the parameters for querying the tweets using API.
    param cwd: current working directory
    return: fields.
    """
    config_dir = os.path.join(cwd, 'crawler', 'config', 'fields_expansions')

    with open(os.path.join(config_dir, 'tweets_fields.json')) as file:
        tweets_fields = json.load(file)

    with open(os.path.join(config_dir, 'poll_fields.json')) as file:
        poll_fields = json.load(file)

    with open(os.path.join(config_dir, 'media_fields.json')) as file:
        media_fields = json.load(file)

    with open(os.path.join(config_dir, 'user_fields.json')) as file:
        user_fields = json.load(file)

    with open(os.path.join(config_dir, 'place_fields.json')) as file:
        place_fields = json.load(file)

    with open(os.path.join(config_dir, 'expansions.json')) as file:
        expansions = json.load(file)

    tweets_fields = ','.join(tweets_fields)

    poll_fields = ','.join(poll_fields)

    media_fields = ','.join(media_fields)

    user_fields = ','.join(user_fields)
    # print(user_fields)

    place_fields = ','.join(place_fields)
    # print(place_fields)

    tweets_expansions = ','.join(expansions)
    # print(tweets_expansions)

    return tweets_fields, poll_fields, media_fields, user_fields, place_fields, tweets_expansions
