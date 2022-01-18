import os

import numpy as np
import transformers
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoTokenizer, PfeifferConfig, PfeifferInvConfig, \
    AutoModelForSequenceClassification, AutoModelWithHeads, TrainingArguments
from transformers import TextClassificationPipeline, AdapterConfig, AdapterTrainer, EvalPrediction
from transformers.adapters.composition import Stack, Fuse

from models.scripts.utils.utils import *
from models.scripts.utils.data import *


# fine-tuned on fi/hu/pl
lang = "sv"
# lr_rate = 1e-4
lr_rate = 5e-5
num_train_epochs = 6
logging_steps = 10
batch_size = 64
num_labels = 3
eval_steps = 10
seed = 42

# output_dir = f"output/twitter_xlmr_sentiment/{lang}/"

checkpoint = "cardiffnlp/twitter-xlm-roberta-base"
# checkpoint = "xlm-roberta-base"

output_dir = f"output/twitter_xlmr_sentiment/fusion/{lang}/"
dataset_path = f"datasets/sentiment_analysis/{lang}/preprocessed"
model_name = "twitter_xlmr_sentiment"


# initialize tokenizer and model.
tokenizer, model = load_model_tokenizer(checkpoint, num_labels, heads=True, classify=False)

# loading data
train = load_data(tokenizer, "train", dataset_path, sample_nr=None)
val = load_data(tokenizer, "val", dataset_path, sample_nr=None)
test = load_data(tokenizer, "test", dataset_path, sample_nr=None)


# train an adapter
fi_adapter = get_adapter(model_name, "fi")
hu_adapter = get_adapter(model_name, "hu")
pl_adapter = get_adapter(model_name, "pl")

# mad-x 2.0
# https://docs.adapterhub.ml/adapters.html?highlight=invertible
# invertible adapters are for language adapters.
# add reduction_factor=2
# lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
adapter_config = PfeifferConfig(leave_out=[11])
# load language adapter
print("loading language adapters ...")
# model.load_adapter("de/wiki@ukp", config=lang_adapter_config)

print(f'loading adapter {fi_adapter}')
model.load_adapter(fi_adapter, load_as="fi_senti", config=adapter_config, with_head=False)

print(f'loading adapter {hu_adapter}')
model.load_adapter(hu_adapter, load_as="hu_senti", config=adapter_config, with_head=False)

print(f'loading adapter {pl_adapter}')
model.load_adapter(pl_adapter, load_as="pl_senti", config=adapter_config, with_head=False)

# Add a fusion layer for all loaded adapters
model.add_adapter_fusion(Fuse('fi_senti', 'hu_senti', 'pl_senti'))
model.set_active_adapters(Fuse('fi_senti', 'hu_senti', 'pl_senti'))

# initialize pretrained model and tokenizer.
# sentiment analysis
model.add_classification_head("sa", num_labels=num_labels)

# unfreeze and activate fusion setup
adapter_setup = Fuse('fi_senti', 'hu_senti', 'pl_senti')
model.train_adapter_fusion(adapter_setup)

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

model.save_adapter_fusion(os.path.join(output_dir, "fusion"), "fi,hu,pl")
model.save_all_adapters(os.path.join(output_dir, "fusion"))
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
