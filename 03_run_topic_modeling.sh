#!/bin/sh

# 3.1 build data for inferring topic modeling
# param: lang_code
# define the language code
python -m models.topicModeling.ETM.data_build_for_inferring_topics --lang_code el
# the data would be stored in output/preprocessed/forTP/lang_code/..


# 3.2 infer topics for each language files.
# params: lang_code, model_path, num_topics
# num_topics is indicated by the name of the model path.
# output file: output/results/ETM/lang_code/...
python -m models.topicModeling.ETM.infer_topic_and_filter --lang_code de --num_topics 20


