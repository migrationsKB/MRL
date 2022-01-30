# MigrationsKB
**Multilingual MigrationskB (MGKB)** is a mulitlingual extended version of English MGKB.
The tweets geotagged with Geo location from 32 European Countries
( _Austria, Belgium, Bulgaria, Croatia, Cyprus, Czech, Denmark, Estonia, Finland, France,
Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands,
Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden, Iceland, Liechtenstein, Norway, 
Switzerland, the United Kingdom_ ) are extracted and filtered by 11 languages
(_English, French, Finnish, German, Greek, Dutch, Hungarian, Italian, Polish, Spain, Swedish_). 
Metadata information about the tweets, such as __Geo information (place name, coordinates, country code)__
 are included. **MGKB**  contains sentiments, offensive and hate speeches, topics, hashtags, user mentions in RDF format.
The schema of **MGKB** is an extension of TweetsKB for migration related information. 
Moreover, to associate and represent the potential economic and social factors driving 
the migration flows, the data from Eurostat  and FIBO ontology was used. To represent 
multilinguality, the CIDOC Conceptual Reference Model (CIDOC-CRM) is used. 
The extracted economic indicators, i.e., GDP Growth Rate, Total Unemployment Rate, 
Youth Unemployment Rate, Long-term Unemployment Rate and Income per househould,
are connected with each tweet in RDF using geographical and temporal dimensions. 


Please contact **Yiyi Chen (yiyi.chen@partner.kit.edu)** for pretrained models (Sentiment analysis/hate speech detection/ETM) if necessary.

 
## Resources

MGKB TTL files and topic words in 11 Languages : https://zenodo.org/record/5918508 


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

