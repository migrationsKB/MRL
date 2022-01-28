import argparse

import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import classification_report
from transformers import AdapterTrainer, PfeifferConfig, PfeifferInvConfig, TrainingArguments, EvalPrediction

from models.scripts.utils.data import *
from models.scripts.utils.utils import *

parser = argparse.ArgumentParser(description='Fine tuning xlmr modeling with SA or HSD')
parser.add_argument('--lang_code', type=str, default="en", help="The language of the dataset")
parser.add_argument('--task', type=str, default="sa", help="sa or hsd")
parser.add_argument('--checkpoint', type=str, default="cardiffnlp/twitter-xlm-roberta-base", help="model from huggingface..")
args = parser.parse_args()

lang = args.lang_code
lr_rate = 1e-4
num_train_epochs = 6
logging_steps = 20
batch_size = 8  # SERVER

eval_steps = 100
seed = 42

# output_dir = f"output/twitter_xlmr_sentiment/{lang}/"
# output_dir = f"output/xlmr_sentiment/{lang}/"
# checkpoint = "cardiffnlp/twitter-xlm-roberta-base"
# checkpoint = "xlm-roberta-base"

# output_dir = f"output/twitter_xlmr_hatespeech/{lang}/"
if args.task == "sa":
    num_labels = 3
    output_dir = f"output/twitter_xlmr_sentiments/{lang}/"
    dataset_path = f"datasets/sentiment_analysis/{lang}/preprocessed"
    adapter_name = f"sentiment-{lang}"
    print(f"dataset path: {dataset_path}")
elif args.task == "hsd":
    num_labels = 2
    output_dir = f"output/xlmr_hatespeech/{lang}/"
    dataset_path = f"datasets/hate_speech_detection/{lang}/preprocessed"
    adapter_name = f"hsd-{lang}"
    print(f"dataset path: {dataset_path}")
else:
    num_labels = None
    output_dir = None
    dataset_path = None
    adapter_name = None
    print('Task can either be sentiment analysis (sa) or hate speech detection (hsd)')

# https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
# checkpoint = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# dataset_path = f"datasets/sentiment_analysis/hu/preprocessed/"
# dataset_path = f"datasets/XED/preprocessed/{lang}/"
# try:
# dataset_path = f"datasets/sentiment_analysis/{lang}/preprocessed"
# dataset_path = f"datasets/hate_speech_detection/HateXplain/preprocessed/"


# fine-tune either model.
# https://colab.research.google.com/drive/1IAA1h8u53O1hi9807u7oOFuT3728N0-n?usp=sharing

# datasets:
# https://huggingface.co/datasets/xed_en_fi
# https://huggingface.co/DaNLP/da-xlmr-ned
# dutch : dutch_social
# migration movement data: https://zenodo.org/record/2536590
# initialize tokenizer and model.
tokenizer, model = load_model_tokenizer(args.checkpoint, num_labels, heads=False, classify=True)

# loading data
train = load_data(tokenizer, "train", dataset_path, sample_nr=None)
val = load_data(tokenizer, "val", dataset_path, sample_nr=None)
test = load_data(tokenizer, "test", dataset_path, sample_nr=None)

# train an adapter
print('adapter name: ', adapter_name)
# mad-x 2.0
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
    logging_dir=os.path.join(output_dir, "logs"),
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
    eval_dataset=val,
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
report = classification_report(test_labels, test_preds, digits=3)
print(report)

with open(os.path.join(output_dir, "results.txt"), 'w') as f:
    f.write(str(report))
