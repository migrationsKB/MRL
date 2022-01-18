import os

import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle


def load_data_df(dataset, dataset_path, sample_nr):
    """
    Load data from dataset for fine-tuning.
    :param dataset:
    :param dataset_path:
    :param sample_nr:
    :return:
    """
    df = pd.read_csv(os.path.join(dataset_path, f"{dataset}.csv"))
    df.label = df.label.astype(int)
    if sample_nr:
        df = df.sample(sample_nr)
    print(dataset)
    print(df['label'].value_counts())
    df = shuffle(df)
    return df


def load_data(tokenizer, dataset_name, dataset_path, sample_nr=None):
    df = load_data_df(dataset_name, dataset_path, sample_nr)
    data_set = Dataset.from_pandas(df)

    def tokenize_function(examples):
        # to solve the error of stack dim, define max_length.
        return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)
        # return tokenizer(examples["text"], padding=True, truncation=True)

    data_set = data_set.map(tokenize_function, batched=True)
    data_set.rename_column_("label", "labels")

    # Transform to pytorch tensors and only output the required columns
    data_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return data_set


if __name__ == '__main__':
    dataset = "train"
    dataset_path = "datasets/sentiment_analysis/sv/preprocessed"
    df = load_data_df(dataset, dataset_path, sample_nr=None)


