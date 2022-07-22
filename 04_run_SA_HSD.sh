#!/bin/sh

# 4.1 sentiment analysis
python -m models.scripts.inference --lang_code el --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment --use_adapter True
python -m models.scripts.inference --lang_code fi --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment --use_adapter True
python -m models.scripts.inference --lang_code hu --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment --use_adapter True
python -m models.scripts.inference --lang_code nl --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment --use_adapter True
python -m models.scripts.inference --lang_code pl --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment --use_adapter True
python -m models.scripts.inference --lang_code sv --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment --use_adapter True

python -m models.scripts.inference --lang_code de --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment
python -m models.scripts.inference --lang_code en --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment
python -m models.scripts.inference --lang_code fr --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment
python -m models.scripts.inference --lang_code it --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment
python -m models.scripts.inference --lang_code es --task sa --checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment


# 4.2 hate speech analysis
python -m models.scripts.inference --lang_code el --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code fi --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code hu --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code nl --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code pl --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code sv --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code de --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code en --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code fr --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code it --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
python -m models.scripts.inference --lang_code es --task hsd --checkpoint cardiffnlp/twitter-xlm-roberta-base --use_adapter True
