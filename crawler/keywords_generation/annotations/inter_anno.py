import os
import pandas as pd


def exchange_anno(lang='de'):
    anno_zero = '00'
    anno_one = '01_11langs'  # automatic
    anno_zero_file = os.path.join(anno_zero, lang + '.csv')
    anno_one_file = os.path.join(anno_one, lang + '.csv')

    save_dir = "final"
    anno_zero_df = pd.read_csv(anno_zero_file, header=None)
    anno_one_df = pd.read_csv(anno_one_file, sep=';')
    print(anno_one_df.columns)
    anno_one_keywords = anno_one_df[anno_one_df['Annotation '] == 1]['keyword']
    anno_zero_keywords = list(set(anno_zero_df[0].str.lower()))
    print(len(anno_one_keywords))
    anno_one_keywords = list(set(anno_one_keywords.str.lower()))
    print(len(anno_one_keywords), anno_one_keywords)

    print(len(anno_zero_keywords))
    print(set(anno_zero_keywords).difference(set(anno_one_keywords)))
    print(set(anno_one_keywords).intersection(set(anno_zero_keywords)))


def inspect_keywords(lang='de'):
    anno_one = '02'  # automatic
    anno_one_file = os.path.join(anno_one, lang + '.csv')
    anno_one_df = pd.read_csv(anno_one_file, sep=',')
    if lang != "en":
        if 'deepl' in anno_one_df.columns:
            anno_one_df = anno_one_df[anno_one_df['Annotation'] == 1][['keyword', 'deepl']]
        else:
            anno_one_df = anno_one_df[anno_one_df['Annotation'] == 1][['keyword', 'google']]
    else:
        anno_one_df = anno_one_df[anno_one_df['Annotation'] == 1]['keyword']

    save_dir = "final"
    save_file = os.path.join(save_dir, lang + '.csv')
    anno_one_df.to_csv(save_file, index=False)


if __name__ == '__main__':
    # for lang in ['en', 'fi', 'fr', 'de', 'el', 'nl', 'hu', 'ga', 'it', 'pl', 'es', 'sv']:
    for lang in ['bg', 'hr', 'cs', 'da', 'et', 'lv', 'lt', 'mt', 'pt', 'ro', 'sk', 'sl', 'no', 'is']:
        print(lang)
        inspect_keywords(lang=lang)
