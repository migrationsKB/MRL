import json
import os

import pandas as pd
from pandarallel import pandarallel

from utils.utils import timing

pandarallel.initialize()


def get_long(geo):
    if str(geo) != 'nan':
        if 'coordinates' in geo:
            return geo['coordinates']['coordinates'][0]
        else:
            return None
    else:
        return None


def get_lat(geo):
    if str(geo) != 'nan':
        if 'coordinates' in geo:
            return geo['coordinates']['coordinates'][1]
        else:
            return None
    else:
        return None


def get_place_id(geo):
    if str(geo) != 'nan':
        if 'place_id' in geo:

            return geo['place_id']
        else:
            return None
    else:
        return None


def get_hashtags(entities):
    if str(entities) != 'nan':
        if 'hashtags' in entities:
            return [x['tag'] for x in entities['hashtags']]
        else:
            return None
    else:
        return None


def get_mentions(entities):
    if str(entities) != 'nan':
        if 'mentions' in entities:
            return [x['username'] for x in entities['mentions']]
        else:
            return None
    else:
        return None


def get_retweet(public_metrics):
    return public_metrics['retweet_count']


def get_like(public_metrics):
    return public_metrics['like_count']


def get_reply(public_metrics):
    return public_metrics['reply_count']


##########################################

@timing
def dic2df(tweet_dict, country_code):
    tweets = tweet_dict['data']
    places = tweet_dict['places']
    # get geo information
    place_df = pd.DataFrame.from_dict(places, orient='index')
    place_df.rename(columns={'id': 'place_id'}, inplace=True)
    # tweets data df
    df = pd.DataFrame.from_dict(tweets, orient='index')

    # coordinates
    df['long'] = df['geo'].parallel_apply(get_long)
    df['lat'] = df['geo'].parallel_apply(get_lat)

    # prepare to use place_df
    df['place_id'] = df['geo'].parallel_apply(get_place_id)

    # hashtags, user mentions
    df['hashtags'] = df['entities'].parallel_apply(get_hashtags)
    df['user_mentions'] = df['entities'].parallel_apply(get_mentions)

    # user interactions
    df['reply_count'] = df['public_metrics'].parallel_apply(get_reply)
    df['like_count'] = df['public_metrics'].parallel_apply(get_like)
    df['retweet_count'] = df['public_metrics'].parallel_apply(get_retweet)

    # merge df and place_df
    merged = pd.merge(df, place_df, left_on='place_id', right_on='place_id', how='left')
    merged.rename(columns={'geo_y': 'geo'}, inplace=True)
    columns_reserved = ['author_id', 'conversation_id', 'text',
                        'id', 'created_at', 'lang',
                        'long', 'lat',
                        'hashtags', 'user_mentions',
                        'reply_count', 'like_count',
                        'retweet_count',
                        'full_name', 'name', 'country', 'geo',
                        'country_code']
    merged = merged[columns_reserved]
    merged['country_code'] = [country_code for x in range(len(merged))]
    return merged


@timing
def dict2df_one_file(input_file, outfile):
    with open(input_file) as reader:
        tweet_dict = json.load(reader)
    df_ls = []
    for country_code, tweet_dict in tweet_dict.items():
        df_country = dic2df(tweet_dict, country_code)
        print(country_code, len(df_country))
        df_ls.append(df_country)

    df = pd.concat(df_ls)
    df.index = df.id
    print(len(df))
    df.to_csv(outfile)


if __name__ == '__main__':
    # countries: ['GB', 'DE', 'SE', 'FR', 'IT', 'GR', 'ES', 'AT', 'HU', 'CH', 'PL', 'NL']
    # langs: ['en', 'fi', 'fr', 'de', 'el', 'nl', 'hu', 'ga', 'it', 'pl', 'es', 'sv']
    input_dir = 'output/preprocessed/restructured'
    out_dir = 'output/preprocessed/csv/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            country_lang_code = filename.replace('.json', '')
            country_, lang_ = country_lang_code.split('-')
            print(country_, lang_)
            # check the language directories
            lang_dir = os.path.join(out_dir, lang_)
            if not os.path.exists(lang_dir):
                os.mkdir(lang_dir)

            infile_path = os.path.join(input_dir, filename)
            with open(infile_path) as f:
                data = json.load(f)

            if len(data[country_]['data']) > 0:
                df_ = dic2df(data[country_], lang_)
                save_file_path = os.path.join(lang_dir, f"{country_}-{lang_}.csv")
                df_.to_csv(save_file_path)

    # test_file = 'output/preprocessed/restructured/DE-fi.json'
    # with open(test_file) as f:
    #     data = json.load(f)
    # df_de = dic2df(data['DE'], 'DE')
    # df_de.to_csv('output/preprocessed/DE-fi.csv')
