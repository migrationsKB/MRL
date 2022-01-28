# Multilingual MigrationsKB

## Overall Framework


### Collect tweets
- get Twitter api and put `credentials.yaml` in `crawler/config` folder
  - ```
    migrationsKB:
       berear_token: XXXX
    ```
- specify the `COUNTRY_ISO2`, and `idx` of `keywords_all`
  - run `python -m crawler.main_keywords`

### Preprocessing tweets

1. restructure data and get statistics of curated data by country
    - `python -m preprocessor.restructure_data`

### Topic Modeling

`
python -m models.topicModeling.ETM.main --mode train --num_topics 50  --lang_code es
`

* Steps:

```
1. data_build_tweets.py
2. skipgram.py
3. python -m models.topicModeling.ETM.main --mode train --num_topics 50 --lang_code es
    * train in batch
        python -m run_etms --min_topics 5 --max_topics 50 --device 0 --lang_code en
4. python -m models.topicModeling.ETM.infer_topic_and_filter --lang_code fi 
--model_path output/models/ETM/fi/best/etm_tweets_K_10_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0_val_loss_6.446055066569226e+18_epoch_188 
--num_topics 10
```


### XLM-R

* fine-tune xlm-r with sentiment analysis or hate speech detection
```
python -m models.scripts.xlm-r-adapter --lang_code swedish --task hsd
```

`
CUDA_VISIBLE_DEVICES=1 python -m models.scripts.xlm-r-adapter --lang_code swedish --task hsd
`

`
CUDA_VISIBLE_DEVICES=3 python -m models.scripts.xlm-r-adapter --lang_code sv --task sa
`


## Trouble shooting
sentencepiece in mac:
https://github.com/google/sentencepiece/issues/378#issuecomment-969896519

