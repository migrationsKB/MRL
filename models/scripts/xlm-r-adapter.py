import os

import numpy as np
import transformers
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoModelWithHeads, AutoTokenizer, TrainingArguments, AdapterTrainer, \
    EvalPrediction, PfeifferConfig, PfeifferInvConfig, AutoModelForSequenceClassification

lang = "fi"
lr_rate = 1e-4
num_train_epochs = 6
logging_steps = 10
batch_size = 64
num_labels = 3
eval_steps = 20
seed = 42

output_dir = f"output/twitter_xlmr_sentiment/{lang}/"
# checkpoint = "cardiffnlp/twitter-xlm-roberta-base"

# https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
checkpoint = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# dataset_path = f"datasets/sentiment_analysis/hu/preprocessed/"
dataset_path = "datasets/XED/preprocessed/fi/"


# fine-tune either model.
# https://colab.research.google.com/drive/1IAA1h8u53O1hi9807u7oOFuT3728N0-n?usp=sharing

# datasets:
# https://huggingface.co/datasets/xed_en_fi
# https://huggingface.co/DaNLP/da-xlmr-ned
# dutch : dutch_social
# migration movement data: https://zenodo.org/record/2536590

def load_data(dataset, dataset_path, sample_nr):
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
    return df


train_df = load_data("train", dataset_path, None)
dev_df = load_data("val", dataset_path, None)
test_df = load_data("test", dataset_path, None)

# https://huggingface.co/docs/datasets/loading_datasets.html
train = Dataset.from_pandas(train_df)
dev = Dataset.from_pandas(dev_df)
test = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


def tokenize_function(examples):
    # to solve the error of stack dim, define max_length.
    return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)
    # return tokenizer(examples["text"], padding=True, truncation=True)


train = train.map(tokenize_function, batched=True)
dev = dev.map(tokenize_function, batched=True)
test = test.map(tokenize_function, batched=True)

# The transformers model expects the target class column to be named "labels"
train.rename_column_("label", "labels")
dev.rename_column_("label", "labels")
test.rename_column_("label", "labels")

# Transform to pytorch tensors and only output the required columns
train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dev.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
# model = AutoModelWithHeads.from_pretrained(checkpoint, config=config)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config= config)

# train an adapter
adapter_name = f"sentiment-{lang}"
## mad-x 2.0
# https://docs.adapterhub.ml/adapters.html?highlight=invertible
# invertible adapters are for language adapters.
adapter_config = PfeifferConfig(leave_out=[11])
model.add_adapter(adapter_name, config=adapter_config)
# model.add_classification_head(
#     adapter_name,
#     num_labels=3)
model.train_adapter(adapter_name)

training_args = TrainingArguments(
    learning_rate=lr_rate,
    num_train_epochs=num_train_epochs,
    logging_dir= os.path.join(output_dir, "logs"),
    logging_steps=logging_steps,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir=output_dir,
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
    seed=seed,
    load_best_model_at_end=True,
    do_eval=True,
    eval_steps=eval_steps,
    evaluation_strategy="steps"
)

# global val_history
val_history = []


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='macro')
    re = recall_score(p.label_ids, preds, average='macro')
    acc = (preds == p.label_ids).mean()
    val_history.append(f1)
    return {"macro_f1": f1, "macro_recall": re, "acc": acc}


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=dev,
    compute_metrics=compute_accuracy,
)

print('start training adapter ....')
trainer.train()

print('start evaluating ....')
trainer.evaluate()

model.save_adapter(os.path.join(output_dir, adapter_name), adapter_name)

# best_step = val_history.index(val_history.pop()) + 1
# n_steps = len(val_history)
print('validation history:', val_history)
# print('best step:', best_step)

# save adapter.

# --- EVALUATION ---

# test_labels = dataset["test"]["labels"]
test_preds_raw, test_labels, out = trainer.predict(test)
test_preds = np.argmax(test_preds_raw, axis=-1)

print(out)
test_preds_raw_path = os.path.join(output_dir, f"preds_test.txt")
np.savetxt(test_preds_raw_path, test_preds_raw)

print(classification_report(test_labels, test_preds, digits=3))
