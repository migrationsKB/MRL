import os
import time
from functools import reduce

import pandas as pd

lang_dict = {"de": "german", "nl": "dutch", "en": "english", "fi": "finnish",
             "fr": "french", "el": "greek", "hu": "hungarian", "it": "italian",
             "pl": "polish", "es": "spanish", "sv": "swedish"}

langs = ["de", "nl", "en", "fi", "fr", "el", "hu", "it", "pl", "es", "sv"]


def put_files_by_language(input_dir="output/preprocessed/csv/", output_dir="output/by_lang/"):
    columns = ["author_id", "conversation_id", "id", "created_at", "lang", "long", "lat",
               "hashtags", "user_mentions", "reply_count", "like_count", "retweet_count",
               "full_name", "name", "country", "country_code", "country_iso2"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for lang in langs:
        lang_dfs = []
        for file in os.listdir(os.path.join(input_dir, lang)):

            filepath = os.path.join(input_dir, lang, file)
            print(filepath)
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath, lineterminator='\n')
                if len(df) > 0:
                    country_code, lang = file.replace(".csv", "").split("-")
                    df["country_iso2"] = [country_code for x in range(len(df))]

                    lang_dfs.append(df[columns])
        df_merge = pd.concat(lang_dfs)
        print(lang, len(df_merge))
        df_merge.to_csv(os.path.join(output_dir, f"{lang}.csv"), index=False)
    print(f"Done putting files by language to {output_dir}!")


def merge_semantic_analysis_results(lang_dir="output/by_lang",
                                    hsd_dir="output/results/HSD",
                                    sa_dir="output/results/SA",
                                    output_dir="output/results/merged/"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for lang in langs:
        lang_file = os.path.join(lang_dir, f"{lang}.csv")  # id
        hsd_file = os.path.join(hsd_dir, f"{lang_dict[lang]}.csv")  # id
        sa_file = os.path.join(sa_dir, f"{lang}.csv")  # id
        print(lang_file, hsd_file, sa_file)
        lang_df = pd.read_csv(lang_file)
        # change the column name for HSD files.
        hsd_df = pd.read_csv(hsd_file).rename(columns={"sentiment": "hsd"})
        print(hsd_df.hsd.value_counts())
        sa_df = pd.read_csv(sa_file)
        dfs = [lang_df, hsd_df, sa_df]
        df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer",
                                                        suffixes=('', '_drop')), dfs)
        df_merged.drop([col for col in df_merged.columns if 'drop' in col], axis=1, inplace=True)
        df_merged.to_csv(os.path.join(output_dir, f"{lang}.csv"))


def main(lang_dir="output/by_lang", output_dir="output/results/merged/"):
    put_files_by_language(output_dir=lang_dir)

    time.sleep(2)

    merge_semantic_analysis_results(lang_dir=lang_dir,output_dir=output_dir)


if __name__ == '__main__':
    import plac
    plac.call(main)
