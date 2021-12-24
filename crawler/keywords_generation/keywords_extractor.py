import os
from datetime import datetime

import yaml
import rootpath
import pandas as pd
import fasttext
import fasttext.util


def load_seedwords(rootdir):
    seedword_dict = yaml.load(open(os.path.join(rootdir, 'crawler',  'keywords_generation', 'seed.yml')), yaml.FullLoader)
    return seedword_dict


def extract_keywords_per_lang(words, lang='es', model_name='fasttext'):
    if model_name == "fasttext":
        model_dir = '/Users/yiyichen/Documents/wordEmbeddings/fastText/'
        model_path = os.path.join(model_dir, f'cc.{lang}.300.bin')
        if os.path.exists(model_path):
            print(f'model exists {model_path}')
            ft = fasttext.load_model(model_path)

            keywords = []
            for word in words:
                print('seed word:', word)
                nns = [x for _, x in ft.get_nearest_neighbors(word, k=50)]
                keywords += nns

            dedup_keywords = list(set(keywords))
            return dedup_keywords


def extract_keywords(output_dir='crawler/keywords_generation/keywords_fasttext'):
    # detect the root directory
    rootdir = rootpath.detect()

    seedwords_dict = load_seedwords(rootdir)
    print(seedwords_dict)

    for lang in ['no', 'is']:
    # for lang, words in seedwords_dict.items():
        words = seedwords_dict[lang]
        starttime = datetime.now()
        print(f'language: {lang}')
        dedup_keywords = extract_keywords_per_lang(words, lang=lang)
        print('number of keywords_generation extracted: ', len(dedup_keywords))

        # save to csv file.
        df = pd.DataFrame(dedup_keywords)
        filepath: str = os.path.join(output_dir, lang + '.csv')
        df.to_csv(filepath, header=False, index=False)

        endtime = datetime.now()

        print(f'Working time: {endtime - starttime}')
        print('*' * 40)


if __name__ == '__main__':
    extract_keywords()
