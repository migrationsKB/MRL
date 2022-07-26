import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

label_map = {
    "emotion": {"1": "ANGER", "2": "ANTICIP.", "3": "DISGUST", "4": "FEAR", "5": "JOY", "6": "SADNESS", "7": "SURPRISE",
                "8": "TRUST"},
    "sentiment": {"1": "negative", "2": "positive", "3": "negative", "4": "negative", "5": "positive", "6": "negative",
                  "7": "unknown", "8": "positive"}
}

config = yaml.load(open(os.path.join('datasets', 'config.yaml')), yaml.FullLoader)
TRAIN_LEN = config["xlm-t"]["train_len"]
VAL_LEN = config["xlm-t"]["val_len"]
TEST_LEN = config["xlm-t"]["test_len"]
ALL_LEN = config['xlm-t']['LEN']
print(ALL_LEN)


def read_data(data_path, lang):
    # load_data
    df = pd.read_csv(data_path, sep='\t', header=None).rename({
        0: "text", 1: "emotions"
    }, axis=1)

    df["emotions"] = df["emotions"].str.replace(" ", "")

    # Create one vs all Encoding
    for i in range(1, 9):
        df[str(i)] = df["emotions"].str.contains(str(i)).astype(int).replace(1, label_map["sentiment"][str(i)])
        df[label_map["emotion"][str(i)]] = df["emotions"].str.contains(str(i)).astype(int)

    # Calculate number of positive and number of negative samples in the dataset
    df["n_positive"] = np.sum((df.iloc[:, 0:18] == "positive"), axis=1)
    df["n_negative"] = np.sum((df.iloc[:, 0:18] == "negative"), axis=1)
    df = df.drop([str(i) for i in range(1, 9)], axis=1)
    
    # How many annotations are there per sample?
    df["n_annots"] = np.sum(df.iloc[:, 2:10], axis=1)
    if lang in ["en", "fi"]:
        
        # Determine if the annotation is more positive or negative
        df = df[df['n_annots'] == 1]

        df.loc[df['n_negative'] == 1, "sentiment"] = "negative"
        df.loc[df['n_positive'] == 1, "sentiment"] = "positive"
    else:
        #Calculate rel_pos_scores
        df["rel_pos"] = df["n_positive"] / (df["n_positive"] + df["n_negative"])
        df["rel_neg"] = df["n_negative"] / (df["n_positive"] + df["n_negative"])
        print(df.rel_pos.value_counts())

        df.loc[df["rel_pos"] >= 1.0, "sentiment"] = "positive"
        df.loc[(df["rel_pos"] == 0.5), "sentiment"] = "neutral"

        df.loc[df["rel_pos"] <= 0.0, "sentiment"] = "negative"
        df = df[~df["sentiment"].isna()]

    df.drop_duplicates(subset='text', inplace=True)
    print('after dedup, ', len(df))
    print(df.sentiment.value_counts())
    # return df[['text', 'sentiment']]
    return df

def read_txt(filepath, lang):
    texts = []
    with open(filepath) as f:
        for line in f.readlines():
            if line.split("\t"):
                if lang == "fi":
                    _, text, _, _, _ = line.split('\t')
                else:
                    _, text = line.split("\t")
                texts.append(text.replace('\n', ''))
    neus = ["neutral" for _ in range(len(texts))]
    df = pd.DataFrame.from_dict({"text": texts, "sentiment": neus})
    df.drop_duplicates(subset="text", inplace=True)
    return df


def preprocess_fi_en_data():
    lang = "fi"
    data_dir = "datasets/XED"
    save_dir = os.path.join(data_dir, "preprocessed")
    save_path = os.path.join(save_dir, lang)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    data_path = os.path.join(data_dir, "AnnotatedData", f"{lang}-annotated.tsv")
    neu_data_path = os.path.join(data_dir, "AnnotatedData", f"neu_{lang}.txt")

    df_neu = read_txt(neu_data_path, lang=lang)
    print(df_neu)
    df_pos_neg = read_data(data_path, lang)
    df = pd.concat([df_neu, df_pos_neg])
    df.rename(columns={'sentiment': 'label'}, inplace=True)
    df['label'].replace(to_replace=["negative", "neutral", "positive"], value=[0, 1, 2], inplace=True)
    df['LEN'] = df.text.str.len()  # median:30, max:281, min:1
    df = df[df['LEN'] >= 30]
    # fi: 30 , en:50
    print(df['label'].value_counts())
    df_neutral = df[df['label'] == 1].sample(1011, random_state=1)
    df_positive = df[df['label'] == 2].sample(1011, random_state=1)
    df_negative = df[df['label'] == 0].sample(1011, random_state=1)
    df_ = pd.concat([df_neutral, df_negative, df_positive])
    df_ = df_[['text', 'label']]
    train, test_dev = train_test_split(df_, test_size=(TEST_LEN+VAL_LEN)/ALL_LEN)
    val, test = train_test_split(test_dev, test_size=TEST_LEN/(len(test_dev)))
    print(len(train), len(test), len(val))
    print(TRAIN_LEN, TEST_LEN, VAL_LEN)
    train.to_csv(os.path.join(save_path, "train.csv"), index=False)
    test.to_csv(os.path.join(save_path, "test.csv"), index=False)
    val.to_csv(os.path.join(save_path, "val.csv"), index=False)


def create_multilingual_data(lang="sv"):
    data_dir = "datasets/XED"
    save_dir = os.path.join('datasets/sentiment_analysis', lang, "preprocessed")
    projection_dir = os.path.join(data_dir, "Projections")
    # languages = ["el", "nl", "pl", "sv"]
    languages = [lang]
    for lang in languages:
        save_path = os.path.join(save_dir)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        datapath = os.path.join(projection_dir, f"{lang}-projections.tsv")
        df = read_data(datapath, lang)
        print(df.sentiment.value_counts())
        df.rename(columns={'sentiment': 'label'}, inplace=True)
        df['label'].replace(to_replace=["negative", "neutral", "positive"], value=[0, 1, 2], inplace=True)
        df['LEN'] = df.text.str.len()  # median:30, max:281, min:1
        print(max(df['LEN']), np.median(df['LEN']), min(df['LEN']))
        df = df[df['LEN'] > 10]
        # el:10
        print(df.label.value_counts())
        df_neutral = df[df['label'] == 1]
        df_positive = df[df['label'] == 2]
        df_negative = df[df['label'] == 0]
        if len(df_neutral) < int(ALL_LEN/3):
            df_neutral = df_neutral
            POS_NEG = int(int(ALL_LEN - len(df_neutral))/2)
            df_positive = df_positive.sample(POS_NEG, random_state=1)
            df_negative = df_negative.sample(POS_NEG, random_state=1)
        else:
            df_neutral = df_neutral.sample(1011, random_state=1)
            df_positive = df_positive.sample(1011, random_state=1)
            df_negative = df_negative.sample(1011, random_state=1)

        df_ = pd.concat([df_neutral, df_negative, df_positive])
        print(df_.label.value_counts())
        df_ = df_[['text', 'label']]
        train, test_dev = train_test_split(df_, test_size=(TEST_LEN + VAL_LEN) / ALL_LEN)
        val, test = train_test_split(test_dev, test_size=TEST_LEN / (len(test_dev)))
        print(len(train), len(test), len(val))
        print(TRAIN_LEN, TEST_LEN, VAL_LEN)
        train.to_csv(os.path.join(save_path, "train.csv"), index=False)
        test.to_csv(os.path.join(save_path, "test.csv"), index=False)
        val.to_csv(os.path.join(save_path, "val.csv"), index=False)


if __name__ == '__main__':
    # create_multilingual_data()
    data_dir = "datasets/XED"
    lang = "sv"
    save_dir = os.path.join('datasets/sentiment_analysis', lang, "preprocessed")
    projection_dir = os.path.join(data_dir, "Projections")

    datapath = os.path.join(projection_dir, f"{lang}-projections.tsv")
    df = read_data(datapath, lang)
    print(df[df['sentiment']=="neutral"]['text'].tolist())
    print(df[df['sentiment']=="positive"]['text'].tolist())

    print(df[df['sentiment']=="negative"]['text'].tolist())

