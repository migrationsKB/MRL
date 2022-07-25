# -----------------------------------------------
# Crawling tweets using Twitter API (academic research, v2, full archive search)
# 2021 Yiyi Chen, Karlsruhe, Germany
# Project: ITFlows, EU Horizon
# Released under a Creative Commons Attribution 4.0 International License.
# email: chen.yiyi@pm.me
# -----------------------------------------------
import os
import gzip
from datetime import datetime
import json
from glob import glob
from collections import OrderedDict
import time
from typing import Any

import requests
import numpy as np
import pandas as pd

from utils.api_authen import load_academic_research_bearer
from utils.utils import get_params, timing
from utils.logger import logger

logger(output_file=f"{datetime.now()}.log")

cwd = os.getcwd()


@timing
def get_last_start_time(dir_path: str) -> Any:
    """
    Get the last earliest crawled dates for tweets as the end time for next round of crawling
    :param dir_path: path to the directory
    :return: str or None
    """
    files = glob(dir_path + '/**.gz')
    print('nr of existing files:', len(files))
    if files is not None and files != []:
        print(files)
        dir_dict = {int(filepath.split('_')[1]): filepath for filepath in files}
        od = OrderedDict(sorted(dir_dict.items(), reverse=True))
        first_key = list(od)[0]
        start_time = od[first_key].split('_')[2].replace('.gz', '')
        print('last start time:', start_time)
        return start_time
    else:
        return None


@timing
def query_main(api_name, batch_idx, country_iso2, keywords_list, idx, start_year, end_year, LEN_chunks, lang, Lang_def):
    f"""
    specify hashtag operations
    :param api_name: name of the api
    :param country_iso2: iso2 code for the country
    :param batch_idx: the index of the batch 
    :param keywords_list: list of keywords
    :param idx: the index of the keywords_list
    :param start_year: the start year
    :param end_year: the end year
    :param LEN_chunks: length of keywords chunks
    :param lang: language of the keywords.
    :return:
    """
    # crawling with hashtags categories from first round
    bearer_token = load_academic_research_bearer(cwd, api_name)

    # endpoint for academic research
    search_url = 'https://api.twitter.com/2/tweets/search/all'

    tweets_fields, poll_fields, media_fields, user_fields, place_fields, tweets_expansions = get_params(cwd)

    startdate = start_year + '-01-01T00:00:00.00Z'

    # change end time accordingly.
    # enddate = end_year + '-08-02T00:00:00.00Z'
    # set up start_time and end_time parameters in API call
    # max_results, to maximum 100 when there are special expansions.

    # idx batch of keywords
    keywords = keywords_list[idx]

    # output directory root output/crawled/
    # check if the data dir for a country exists.
    output_dir_root = os.path.join(cwd, 'output', 'crawled', country_iso2)
    if not os.path.exists(output_dir_root):
        os.mkdir(output_dir_root)

    # the other 14 langauges.
    # batch2/lang/idx.
    # output_dir_root_batch = os.path.join(output_dir_root, "batch2")
    output_dir_root_batch = os.path.join(output_dir_root, batch_idx)
    if not os.path.exists(output_dir_root_batch):
        os.mkdir(output_dir_root_batch)

    # output_dir_root_lang = os.path.join(output_dir_root_batch, lang)
    # if not os.path.exists(output_dir_root_lang):
    #     os.mkdir(output_dir_root_lang)

    output_dir_ = os.path.join(output_dir_root_batch, str(idx))

    # output_dir_ = os.path.join(output_dir_root, str(idx))
    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)

    # output_dir = os.path.join(output_dir_, lang)
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    # query, geo country code, no retweets, no promotion.
    if Lang_def == "mix":
        query = "({}) has:geo place_country:{} -is:retweet -is:nullcast " \
                "-lang:en -lang:fi -lang:fr -lang:de -lang:el -lang:nl -lang:hu -lang:it -lang:pl -lang:es " \
                "-lang:sv".format(
            ' OR '.join(keywords), country_iso2, Lang_def)
    elif Lang_def == "und":
        query = "({}) has:geo place_country:{} -is:retweet -is:nullcast " \
                "-lang:en -lang:fi -lang:fr -lang:de -lang:el -lang:nl -lang:hu -lang:it -lang:pl -lang:es " \
                "-lang:sv -lang:cs -lang:da -lang:pt -lang:ro -lang:no ".format(
            ' OR '.join(keywords), country_iso2, Lang_def)
    else:
        query = "({}) has:geo place_country:{} -is:retweet -is:nullcast lang:{}".format(
            ' OR '.join(keywords), country_iso2, Lang_def)
    print(query)
    print('query length => ', len(query))
    assert len(query) <= 1024

    # check the last min id from previous crawling.
    start_time = get_last_start_time(output_dir_)
    if start_time is not None:
        print('the min time from last crawling: ', start_time)

        # https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
        query_params = {'query': query,
                        'tweet.fields': tweets_fields,
                        'user.fields': user_fields,
                        'media.fields': media_fields,
                        'poll.fields': poll_fields,
                        'place.fields': place_fields,
                        'expansions': tweets_expansions,
                        'start_time': startdate, 'end_time': start_time, 'max_results': 100}
    else:
        query_params = {'query': query,
                        'tweet.fields': tweets_fields,
                        'user.fields': user_fields,
                        'media.fields': media_fields,
                        'poll.fields': poll_fields,
                        'place.fields': place_fields,
                        'expansions': tweets_expansions,
                        'start_time': startdate, 'max_results': 100}

    # Updated code: 16.Dec 2021
    # https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Full-Archive-Search/full-archive-search.py
    def bearer_oauth(r):
        """
        Method required by bearer token authentication.
        """

        r.headers["Authorization"] = f"Bearer {bearer_token}"
        r.headers["User-Agent"] = "v2FullArchiveSearchPython"
        return r

    # Query
    # Connect to end point.
    response = requests.request('GET', search_url, auth=bearer_oauth, params=query_params)
    print(response.status_code)
    return response, output_dir_, country_iso2


def crawler(api_name, batch_idx, country_iso2, keywords, idx, start_year, end_year, LEN_chunks, lang, LANG_DEF,
            flag=True):
    t = datetime.today().strftime('%Y%m%d%H%M%S')
    response, output_dir, countryiso2 = query_main(api_name, batch_idx, country_iso2, keywords, idx, start_year,
                                                   end_year,
                                                   LEN_chunks, lang, LANG_DEF)
    while flag:

        if response.status_code == 200:

            # data
            try:
                data = response.json()
                print('data output')
                data_json = json.dumps(data) + '\n'
                data_encoded = data_json.encode('utf-8')

                ###########################
                # try:
                df = pd.DataFrame(data['data'])
                dates = df['created_at']
                min_time = np.min(dates)
                print('crawled {} tweets'.format(len(df)))

                # output file path.
                outfile = os.path.join(output_dir, countryiso2 + '_' + t + '_' + str(min_time) + '.gz')

                with gzip.open(outfile, 'w') as outfile:
                    print(f'writing tweets to {outfile}....')
                    outfile.write(data_encoded)

                if flag:
                    time.sleep(5)
                    crawler(api_name, batch_idx, countryiso2, keywords, idx, start_year, end_year, LEN_chunks, lang,
                            LANG_DEF,
                            flag=True)

            except Exception:
                print(f'Exception {Exception}')
                if idx < LEN_chunks:
                    idx += 1
                    print('idx:', idx)
                    time.sleep(5)
                    crawler(api_name, batch_idx, countryiso2, keywords, idx, start_year, end_year, LEN_chunks, lang,
                            LANG_DEF,
                            flag=True)
                break

        # no output
        elif response.status_code == 400:
            print(response.text)
            if idx < LEN_chunks - 1:
                idx += 1
                print('idx:', idx)
                time.sleep(5)
                crawler(api_name, batch_idx, countryiso2, keywords, idx, start_year, end_year, LEN_chunks, lang,
                        LANG_DEF,
                        flag=True)
            else:
                exit()

        # else:
        # response code ==429, rate limit exceeded. 100 api calls finished, wait another 15 minutes.
        # or too many requests
        elif response.status_code == 429:
            if "Rate limit exceeded" in response.text:
                flag = False
                print(response.text)
                Exception(response.status_code, response.text)
                exit()

            else:
                print(response.text)
                output_dir_root = os.path.join(cwd, 'output', 'crawled', country_iso2)
                max_id = sorted([int(x) for x in os.listdir(output_dir_root)])[-1]
                print('max id ', max_id)
                time.sleep(5)
                crawler(api_name, batch_idx, country_iso2, keywords, max_id, start_year, end_year, LEN_chunks, lang,
                        LANG_DEF,
                        flag=True)
        else:
            flag = False
            print(response.text)
            raise Exception(response.status_code, response.text)
            break


def main(COUNTRY_ISO2, batch_idx, start_year, end_year, idx=0, API_NAME="migrationsKB"):
    infile = os.path.join(cwd, 'crawler', 'config', 'keywords', 'all_2.json')

    with open(infile) as f:
        keywords_all = json.load(f)
    print(f"the first chunk of keywords: {keywords_all[0]}")
    print(f"country {COUNTRY_ISO2}")

    LEN = len(keywords_all)
    print(f"length of keywords: {LEN} chunks")
    crawler(API_NAME, batch_idx, COUNTRY_ISO2, keywords_all, idx, str(start_year), str(end_year), LEN, "", "mix")


if __name__ == '__main__':
    # third batch, mixed languages/
    # countries:['GB', 'DE', 'SE', 'FR', 'IT', 'GR', 'ES', 'AT', 'HU', 'CH', 'PL', 'NL']
    #     # 'BE', 'BG', 'CZ', 'DK',
    #     # 'HR', 'CY', 'EE', 'FI',  'IE', 'LV', 'LT', 'LU',  'MT', 'PT', 'RO', 'SK',  'SI',
    #     #  'IS', 'LI', 'NO']
    import plac

    plac.call(main)
