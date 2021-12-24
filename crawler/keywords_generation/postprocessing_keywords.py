import os
import pandas as pd
import numpy as np


def merge2trans(outputdir: str = "crawler/keywords_generation/mergedTrans") -> None:
    deepltrans = 'crawler/keywords_generation/deepLTrans'
    fasttext_dir = 'crawler/keywords_generation/keywords_fasttext'
    googletrans_dir = 'crawler/keywords_generation/googleTrans'

    for filename in os.listdir(fasttext_dir):
        filepath = os.path.join(fasttext_dir, filename)
        print(filepath)
        lang = filename.replace('.csv', '')
        df = pd.read_csv(filepath, header=None)

        deepltrans_file = os.path.join(deepltrans, filename)
        if os.path.exists(deepltrans_file) and lang != 'en':
            googletrans_file = os.path.join(googletrans_dir, filename)
            google_df = pd.read_csv(googletrans_file)

            print(deepltrans_file)

            deepl_df = pd.read_csv(deepltrans_file, header=None)
            df['deepl'] = deepl_df[0]
            df['google'] = google_df['google_translated']
            df.columns = ['keyword', 'deepl', 'google']
            df['deepl==google'] = np.where(df['deepl'] == df['google'], True, False)
            save2file = os.path.join(outputdir, filename)
            df.to_csv(save2file, index=False)


if __name__ == '__main__':
    merge2trans()
