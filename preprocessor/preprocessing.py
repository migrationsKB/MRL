import json
import os
from typing import Any, List, Tuple
import itertools
import re

import pandas as pd
from nltk.tokenize import TweetTokenizer
import unicodedata
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utils.reader import load_config
from utils.utils import timing

tt = TweetTokenizer()

SPECIAL_CHARS = ['&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;', '&cent;', '&pound;', '&yen;', '&euro;',
                 '&copy;', '&reg;']

# https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
# https://github.com/s/preprocessor/blob/master/preprocessor/defines.py
RESERVED_WORDS_PATTERN = re.compile(r'\b(?<![@#])(RT|FAV)\b')
try:
    # UCS-4
    EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
except re.error:
    # UCS-2
    EMOJIS_PATTERN = re.compile(
        u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

SMILEYS_PATTERN = re.compile(r"(\s?:X|:|;|=)(?:-)?(?:\)+|\(|O|D|P|S|\\|\/\s){1,}", re.IGNORECASE)


def get_entities_positions(entities: Any, entity_name: str) -> List[Tuple]:
    """
    Get the positions of the entities
    :param entities: including hashtags, mentions, urls
    :param entity_name: either hashtags, mentions, or urls
    :return: positions of the entities , a list of Tuples.
    """
    positions = []
    if str(entities) != 'nan':
        if entity_name in entities:
            for x in entities[entity_name]:
                positions.append((x['start'], x['end']))
    return positions


def entities_in_text(text, entities, entity_name=""):
    """
    Get the hashtags, urls, or mentions.
    :param text:
    :param entities:
    :param entity_name:
    :return:
    """
    positions = get_entities_positions(entities, entity_name)
    return [text[pos[0]:pos[1]] for pos in positions]


def preprocessing_one_tweet(tweet, forID, remove_emoticons=False, unidecoded=False):
    """
    To preprocess each tweet: remove hashtags (or only "#"), remove urls and give "<USER>"
    unidecoding
    remove html tags
    :param tweet:
    :param forID: if the preprocessing is for language identification.
    :return:
    """
    # for ID, pure text without removing anything
    text = tweet['text']
    # print(text)

    # remove urls, mentions, or hashtags.
    if "entities" in tweet:
        entities = tweet['entities']
        # print(entities)
        hashtags = None
        user_mentions = None
        urls = None
        if "hashtags" in entities:
            hashtags = entities_in_text(text, entities, "hashtags")
        if "mentions" in entities:
            user_mentions = entities_in_text(text, entities, "mentions")
        if "urls" in entities:
            urls = entities_in_text(text, entities, "urls")

        if hashtags is not None:
            for hashtag in hashtags:
                if forID:
                    # language identification, hashtags can be transferable between languages.
                    text = text.replace(hashtag, "")
            text = text.replace("#", '')

        if user_mentions is not None:
            for x in user_mentions:
                text = text.replace(x, "")

        if urls is not None:
            for url in urls:
                text = text.replace(url, "")

    # remove accented characters
    if unidecoded:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove white strings
    text = text.replace('\n', '').replace('\s', '').replace('\r', '')
    # remove special chars, starting with &.
    for CHAR in SPECIAL_CHARS:
        text = text.replace(CHAR, '')

    if remove_emoticons:
        # remove emojis
        text = EMOJIS_PATTERN.sub(r'', text)
        # remove smileys
        text = SMILEYS_PATTERN.sub(r'', text)

    tokenized = tt.tokenize(text)
    if len(tokenized) > 1:
        return text
    else:
        print(text)


def preprocessing_files_by_lang(lang_code, output_dir):
    """
    if lang_code is "und", will use langid mode to preprocess tweets
    :param lang_code:
    :param output_dir:
    :return:
    """
    df = pd.DataFrame()
    count = 0
    # original_texts = []
    preprocessed_texts = []
    # preprocessed_texts_langid = []
    countries = load_config()['countries']
    countries_ = []
    dates = []
    ids = []
    for country in countries:
        test_file = f'output/preprocessed/restructured/{country}-{lang_code}.json'

        with open(test_file) as f:
            data = json.load(f)[country]['data']
        # for tweet_id, tweet in dict(itertools.islice(data.items(), example_nr)).items():
        for tweet_id, tweet in data.items():
            # preprocessed_text_for_langid = preprocessing_one_tweet(tweet, True)
            preprocessed_text = preprocessing_one_tweet(tweet, False)

            if preprocessed_text is not None:
                # print(preprocessed_text)
                # original_texts.append(tweet['text'])
                dates.append(tweet['created_at'])
                preprocessed_texts.append(preprocessed_text)
                # preprocessed_texts_langid.append(preprocessed_text_for_langid)
                ids.append(tweet_id)

                countries_.append(country)

                # print('*' * 40)
            count += 1
    df['id'] = ids
    df['created_at'] = dates
    # df['text'] = original_texts
    df['preprocessed_text'] = preprocessed_texts
    df['country'] = countries_
    # df['preprocessed_text_langid'] = preprocessed_texts_langid
    df.to_csv(os.path.join(output_dir, f'{lang_code}.csv'), index=False)
    print(count)
    print(len(preprocessed_texts))


def main():
    country_code = "DE"
    lang_code = "en"
    example_nr = 10
    output_dir = "output/preprocessed/forBERT"
    preprocessing_files_by_lang(lang_code=lang_code, output_dir=output_dir)


if __name__ == '__main__':
    filepath = os.path.join("output/preprocessed/forBERT", 'en.csv')
    df = pd.read_csv(filepath)
    print(df.head(3))
    # shuffle,
    df = shuffle(df)

    print('LEN: ', len(df))
    df= df.drop_duplicates(subset='preprocessed_text')
    print('LEN: ', len(df))

    print(df.head(3))
    train, val = train_test_split(df, test_size=0.05)
    train_texts = "\n".join(train['preprocessed_text'].tolist())
    val_texts = "\n".join(val['preprocessed_text'].tolist())

    savefile_train = os.path.join("output/preprocessed/forBERT", 'en_train.txt')
    savefile_val = os.path.join("output/preprocessed/forBERT", 'en_val.txt')

    with open(savefile_train, 'w') as file:
        file.write(train_texts)

    with open(savefile_val, 'w') as file:
        file.write(val_texts)
