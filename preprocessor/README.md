# Countries and languages

* [Coding of Countries](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)

# Statistics of Collected Tweets

`python -m preprocessor.stats_analysis`

-> `output/stats/crawled.csv`

* en: 885901/885263 (preprocessed)
* EN: for tp: 885901/874200 (preprocessed)

## from the MigrationsKB 1.0

![](/Users/yiyichen/PycharmProjects/MRL/images/docs/stats_crawled_mgkb1.0.png)

## multilingual lemmatizer and stopwords

stopwords: https://github.com/stopwords-iso/stopwords-iso/tree/master/python

https://blogs.helsinki.fi/language-technology/hi-nlp/morphology/

lemmatizer: https://github.com/adbar/simplemma

following are available:

| Language | Lemmatizer   | alternative                                                     |
|----------|--------------|-----------------------------------------------------------------|
| en       | 0.94 acc     | [lemmInflect]( https://github.com/bjascob/LemmInflect)          |
| fi       | yes          | [voikko](https://voikko.puimula.org/python.html)                |
| fr       | 0.94 acc     |
| de       | 0.95 acc     | [german-nlp](https://github.com/adbar/German-NLP#Lemmatization) |
| el       | low coverage |
| nl       | 0.91 acc     |
| hu       | yes          |
| it       | 0.92 acc     |
| pl       | yes          |
| es       | 0.94 acc     |
| sv       | yes          |
| **prio** |
| bg       | low coverage |
| cs       | low coverage |
| da       | yes          | [lemmy](https://github.com/sorenlind/lemmy)                     |
| et       | low coverage |
| ga       | yes          |
| hr       |
| lv       | yes          |
| lt       | yes          |
| mt       | x (no stopwords)|
| pt       | 0.92 acc     |
| ro       | yes          |
| sk       | 0.87 acc     |
| sl       | low coverage |
| no       | x (nb)       |
| is       | x (no stopwords)|

