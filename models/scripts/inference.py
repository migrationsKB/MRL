import argparse
import torch
import tqdm
from sklearn.metrics import classification_report
from transformers import PfeifferConfig, PfeifferInvConfig, TextClassificationPipeline

from models.scripts.utils.data import *
from models.scripts.utils.utils import *

parser = argparse.ArgumentParser(description='Predicting with xlmr model for SA or HSD')
parser.add_argument('--lang_code', type=str, default="en", help="The language of the dataset")
parser.add_argument('--task', type=str, default="sa", help="sa or hsd")
parser.add_argument('--checkpoint', type=str, default="cardiffnlp/twitter-xlm-roberta-base",
                    help="model from huggingface..")
parser.add_argument("--test_mode", type=bool, default=False, help="whether to test a dataset or to infer...")
# parser.add_argument("--device_nr", type=int, default=0, help="which device of GPU to use")
parser.add_argument("--use_adapter", type=bool, default=False, help="whether to use adapter or not")
args = parser.parse_args()

# define device, either cuda or cpu.
device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device_}")

lang = args.lang_code
lr_rate = 1e-4
num_train_epochs = 6
logging_steps = 10
batch_size = 8
eval_steps = 100
seed = 42

if args.task == "sa":
    num_labels = 3

    if args.use_adapter:
        # adapter_subdir = args.checkpoint.replace("cardiffnlp/", "")
        adapter_dir = f"output/models/SA/{lang}/"
        adapter_name = f"sentiment-{lang}"

        adapter_path = os.path.join(adapter_dir, adapter_name)
        print(f"adapter {adapter_name} from path {adapter_path}")

    if args.test_mode:
        dataset_path = f"datasets/sentiment_analysis/{lang}/preprocessed"
        print(f"dataset path: {dataset_path}")
    else:
        # infer sentiment or hate speech for twitter data
        dataset_path = f"output/results/ETM/{lang}/{lang}_etm.csv"
        print(f"dataset path: {dataset_path}")
elif args.task == "hsd":
    if lang == "en":
        num_labels = 3
    else:
        num_labels = 2
    if args.use_adapter:
        adapter_dir = f"output/models/HSD/{lang}/"
        adapter_name = f"hsd-{lang}"
        adapter_path = os.path.join(adapter_dir, adapter_name)
        print(f"adapter {adapter_name} from path {adapter_path}")

    mapping_dict = {
        "german": "de", "spanish": "es", "dutch": "nl", "hungarian": "hu", "polish": "pl",
        "swedish": "sv", "finnish": "fi", "english": "en", "french": "fr", "italian": "it", "greek": "el"
    }
    # get the path of datasets
    if args.test_mode:
        dataset_path = f"datasets/hate_speech_detection/{lang}/preprocessed"
        print(f"dataset path: {dataset_path}")
    else:
        # infer sentiment or hate speech for twitter data
        lang_ = mapping_dict[lang]
        dataset_path = f"output/results/ETM/{lang_}/{lang_}_etm.csv"
        print(f"dataset path: {dataset_path}")
else:
    num_labels = None
    output_dir = None
    dataset_path = None
    adapter_name = None
    adapter_path = None
    print('Task can either be sentiment analysis (sa) or hate speech detection (hsd)')

# args.checkpoint could be "cardiffnlp/twitter-roberta-base-sentiment", only for English.
# if args.task == "hsd":
tokenizer, model = load_model_tokenizer(args.checkpoint, num_labels, heads=False, classify=True)

# train an adapter
# mad-x 2.0
# https://docs.adapterhub.ml/adapters.html?highlight=invertible
# invertible adapters are for language adapters.
# add reduction_factor=2
# lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)

# load language adapter
# model.load_adapter("de/wiki@ukp", config=lang_adapter_config)

print(f'loading adapter ...')
if args.use_adapter:
    adapter_config = PfeifferConfig(leave_out=[11])
    adapter = model.load_adapter(adapter_path, config=adapter_config)
    # set active adapter
    model.set_active_adapters(adapter)
# model.active_adapters = Stack("de", "hsd-en")

# initialize pretrained model and tokenizer.
# classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework="pt", device=args.device_nr)
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework="pt")

print("evaluating on ", dataset_path)
if args.test_mode:
    test_df = load_data_df("test", dataset_path, None)
    texts = test_df['text'].tolist()
    labels = test_df['label'].tolist()

    predicted = []
    for text, label in zip(texts, labels):
        result = classifier(text)[0]
        predicted.append(int(result['label'].replace("LABEL_", "")))

    clasi_report = classification_report(labels, predicted, digits=3)
    print(clasi_report)
else:
    tweet_df = pd.read_csv(dataset_path)
    texts = tweet_df['preprocessed_cl'].tolist()

    if args.task == "sa":
        predicted = []
        for text in tqdm.tqdm(texts):
            result = classifier(text)[0]
            predicted.append(result['label'])

        save_dir = f"output/results/SA"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        tweet_df["sentiment"] = predicted

        tweet_df.to_csv(os.path.join(save_dir, f"{lang}.csv"), index=False)
        print("values:", tweet_df.sentiment.value_counts())

    if args.task == "hsd":
        predicted = []
        for text in tqdm.tqdm(texts):
            result = classifier(text)[0]
            predicted.append(int(result['label'].replace("LABEL_", "")))

        save_dir = f"output/results/HSD"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # to correct
        tweet_df["sentiment"] = predicted
        tweet_df.to_csv(os.path.join(save_dir, f"{lang}.csv"), index=False)
        print("values:", tweet_df.sentiment.value_counts())

