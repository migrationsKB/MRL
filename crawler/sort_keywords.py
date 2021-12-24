import os
import json

import yaml
import pandas as pd

from utils.utils import load_keywords_for_lang, chunks


def main(cwd=os.getcwd()):
    # config_file = os.path.join(cwd, 'crawler', 'config', 'config.yml')
    # with open(config_file) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # langs = config['langs']

    # keywords_all = []
    keywords_dict = {}
    for lang in ['bg', 'hr', 'cs', 'da', 'et', 'lv', 'lt', 'mt', 'pt', 'ro', 'sk', 'sl', 'no', 'is']:
        # lang_file = os.path.join(cwd, 'crawler', 'config', 'keywords', lang + '.csv')
        lang_file = os.path.join(cwd, 'crawler', 'keywords_generation', 'annotations', 'final', lang+'.csv')
        df = pd.read_csv(lang_file)
        keywords_dict[lang]= list(set(df['keyword'].str.lower().tolist()))
        #keywords_all += df['keyword'].str.lower().tolist()
    #dedup_keywords_all = list(set(keywords_all))
    # keywords_chunks = list(chunks(dedup_keywords_all, n=40))
    # print(keywords_chunks)
    # print(len(keywords_chunks))

    outfile = os.path.join(cwd, 'crawler', 'config', 'keywords', 'all_2_dict.json')

    with open(outfile, 'w') as file:
        json.dump(keywords_dict, file)


def test(filepath):
    with open(filepath) as f:
        keywords = json.load(f)

    print(keywords)


if __name__ == '__main__':
    # cwd = os.getcwd()
    main()
    # outfile = os.path.join(cwd, 'crawler', 'config', 'keywords', 'all_2.json')
    # test(outfile)