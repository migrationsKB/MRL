import os
from typing import Any, Dict
from glob import glob
from collections import defaultdict
import pandas as pd

from utils.reader import read_gz_file, load_config
from utils.utils import timing


@timing
def get_all_languages(data_dir: str):
    """
    Get language statistics of crawled data
    :param data_dir: the data directory to inspect
    :return: a dictionary of language and its number of tweets.
    """
    lang_dict = defaultdict(list)

    for file in glob(data_dir + '/**/**.gz', recursive=True):
        data = read_gz_file(file)
        for tweet in data['data']:
            lang = tweet['lang']
            tweet_id = tweet['id']
            if tweet_id not in lang_dict[lang]:
                lang_dict[lang].append(tweet_id)
    stats_dict = {lang: len(list(set(tweet_ids))) for lang, tweet_ids in lang_dict.items()}
    return stats_dict


@timing
def get_stats_country_lang(stats_file: str, by_country: Any = None, batch=None) -> None:
    """
    Get statistics of crawled tweets
    :param stats_file: filepath to save the statistics
    :param by_country: if it is not None, get statistics by the country
    :return: None
    """
    stats_dict = defaultdict(dict)

    if by_country!=None:
        crawled_data_path = f"output/crawled/{by_country}/{batch}"
        lang_dict = get_all_languages(crawled_data_path)
        stats_dict[by_country] = lang_dict
    else:
        config = load_config()
        print(config)
        countries = config['countries']
        print(countries)
        for country in countries:
            print(country)
            crawled_data_path = f"output/crawled/{country}/"
            lang_dict = get_all_languages(crawled_data_path)
            print(lang_dict)
            stats_dict[country] = lang_dict
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
    stats_df.to_csv(stats_file)


if __name__ == '__main__':
    stats_file_ = os.path.join('output', 'stats', 'crawled_all.csv')
    get_stats_country_lang(stats_file_, None)
