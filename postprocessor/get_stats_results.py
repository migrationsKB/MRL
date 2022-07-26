import json
import os
from collections import defaultdict

import pandas as pd

countries = ['GB', 'DE', 'SE', 'FR', 'IT', 'GR', 'ES', 'AT', 'HU', 'CH', 'PL',
             'NL', 'BE', 'BG', 'CZ', 'DK', 'HR', 'CY', 'EE', 'FI', 'IE', 'LV', 'LT',
             'LU', 'MT', 'PT', 'RO', 'SK', 'SI', 'IS', 'LI', 'NO']

years = [x for x in range(2013, 2022)]


# print(year, label, int(value))
# if label == "normal":
#     label_uri = "https://migrationsKB.github.io/MGKB#normal"
# if label == "offensive":
#     label_uri = "https://migrationsKB.github.io/MGKB#offensive"
#
# w = {"country": {"value": country},
#      "year": {"value": year},
#      "EmotionCategory": {"value": label_uri},
#      "num": {"value": int(value)}}
# normal (0), offensive (1)

def get_result_by_country_and_year(attitude_type, input_dir="output/merged"):
    country_year_dict = defaultdict(dict)
    for country in countries:
        for year in years:
            country_year_dict[country][year] = defaultdict(int)
    print(country_year_dict)

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            filepath = os.path.join(input_dir, file)
            df = pd.read_csv(filepath, low_memory=False)
            print(len(df))
            df.dropna(subset=[attitude_type], inplace=True)
            print(len(df))
            if attitude_type == "sentiment":
                df["sentiment"] = df["sentiment"].replace(
                    {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"})
            if attitude_type == "hsd":
                df["hsd"] = df["hsd"].replace({0: "Normal", 1: "Offensive", 2: "Offensive"})

            df["year"] = df["created_at"].str[:4].astype("int")
            for year, country_code, atti in zip(df["year"], df["country_iso2"], df[attitude_type]):
                if country_code in country_year_dict:
                    if year in country_year_dict[country_code]:
                        country_year_dict[country_code][year][atti] += 1
    if attitude_type == "hsd":
        print(f"hate speech results is written to output/hsd.jsonl")

        with open("output/hsd.jsonl", "w") as file:
            for country in countries:
                for year in years:
                    for label in ["Normal", "Offensive"]:
                        if label == "Normal":
                            label_uri = "https://migrationsKB.github.io/MGKB#normal"
                        if label == "Offensive":
                            label_uri = "https://migrationsKB.github.io/MGKB#offensive"
                        try:
                            value = country_year_dict[country][year][label]
                            w = {"country": {"value": country},
                                 "year": {"value": year},
                                 "EmotionCategory": {"value": label_uri},
                                 "num": {"value": int(value)}}
                            file.write(json.dumps(w) + '\n')
                        except Exception as e:
                            print(e)

    if attitude_type == "sentiment":
        print(f"sentiment results is written to output/sentiment.jsonl")
        with open("output/sentiment.jsonl", "w") as file:
            for country in countries:
                for year in years:
                    for label in ["Negative", "Neutral", "Positive"]:
                        if label == "Negative":
                            label_uri = "http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#negative-emotion"
                        if label == "Neutral":
                            label_uri = "http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#neutral-emotion"
                        if label == "Positive":
                            label_uri = "http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#positive-emotion"

                        try:
                            value = country_year_dict[country][year][label]
                            w = {"country": {"value": country},
                                 "year": {"value": year},
                                 "EmotionCategory": {"value": label_uri},
                                 "num": {"value": int(value)}}
                            file.write(json.dumps(w) + '\n')
                        except Exception as e:
                            print(e)


if __name__ == '__main__':
    import plac
    plac.call(get_result_by_country_and_year)

