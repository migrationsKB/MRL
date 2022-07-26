import os
import json
import pandas as pd
from statistics import mode


def json2df(data, ids):
    texts = []
    labels = []
    for id_ in ids:
        entry = data[id_]
        annots = [x["label"] for x in entry['annotators']]
        texts.append(" ".join(entry['post_tokens']))
        labels.append(mode(annots))
    df = pd.DataFrame.from_dict({"text": texts, "label": labels})
    df['label'].replace(to_replace=["normal", "offensive", "hatespeech"], value=[0, 1, 2], inplace=True)
    df['tweet_id'] = ids
    return df

def read_data(data_dir, save_dir):
    data_path = os.path.join(data_dir, "Data", "dataset.json")
    ids_path = os.path.join(data_dir, "Data", "post_id_divisions.json")

    data = json.load(open(data_path))
    ids = json.load(open(ids_path))

    test_ids = ids["test"]
    train_ids = ids["train"]
    val_ids = ids["val"]

    df_test = json2df(data, test_ids)
    df_val = json2df(data, val_ids)
    df_train = json2df(data, train_ids)

    df_test.to_csv(os.path.join(save_dir, "test.csv"))
    df_val.to_csv(os.path.join(save_dir, "val.csv"))
    df_train.to_csv(os.path.join(save_dir, "train.csv"))


if __name__ == '__main__':
    data_dir = os.path.join("datasets", "hate_speech_detection", "HateXplain")
    save_dir = os.path.join(data_dir, "preprocessed")
    read_data(data_dir, save_dir)
