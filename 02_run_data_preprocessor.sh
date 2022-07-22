#!/bin/sh

# 2.1. Restructure data from .gz -> .json.
# each json file "Country-language.json",
# params: lang, batch-idx, output_folder
# params are optional.
python -m preprocessor.restructure_data


# 2.2 reconstruct json files in to csv files by language.
python -m preprocessor.dict2df

# 2.3 preprocessing tweets for topic modeling and sentiment analysis/hate speech detection
python -m preprocessor.preprocessing

