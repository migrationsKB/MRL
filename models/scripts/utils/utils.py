import os

from transformers import AutoConfig, AutoTokenizer, PfeifferConfig, PfeifferInvConfig, \
    AutoModelForSequenceClassification, AutoModelWithHeads


def load_model_tokenizer(checkpoint, num_labels, heads=False, classify=False):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
    if heads:
        model = AutoModelWithHeads.from_pretrained(checkpoint, config=config)
        return tokenizer, model
    if classify:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
        return tokenizer, model


def get_adapter(model_path, lang):
    adapter_dir = f"output/{model_path}/{lang}/"
    adapter_name = f"sentiment-{lang}"
    adapter_path = os.path.join(adapter_dir, adapter_name)
    print(f" adapter path : {adapter_path}")
    return adapter_path