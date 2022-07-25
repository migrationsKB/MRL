## BERT and masked language model 
Reference: https://docs.adapterhub.ml/training.html

```
export TRAIN_FILE=/path/to/dataset/train
export VALIDATION_FILE=/path/to/dataset/validation

python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir /tmp/test-mlm \
    --train_adapter \
    --adapter_config "pfeiffer+inv"
```


```
python -m models.scripts.run_mlm_adapter --model_name_or_path bert-base-multilingual-uncased --train_file output/preprocessed/forBERT/en_train.txt --validation_file output/preprocessed/forBERT/en_val.txt --output_dir output/models/24-12-2021_15:50_adapters_mlm --line_by_line --do_train --do_eval 
```


To run on own training and validation files, use the following command

```
python run_mlm_adapter.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

**Reference**: https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling

full fine-tuning, without adapters

get version `4.16.0.dev0`
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```


```
ython run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm

```


```angular2html
python -m models.scripts.run_mlm_adapter --model_name_or_path bert-base-multilingual-uncased --train_file output/preprocessed/forBERT/en_train.txt --validation_file output/preprocessed/forBERT/en_val.txt --output_dir output/models/25-12-2021_01:26_adapters --line_by_line --do_train --do_eval --learning_rate 1e-4 --num_train_epochs 10.0 --train_adapter --adapter_config "pfeiffer+inv"
```



```
python models/scripts/run_mlm.py --model_name_or_path bert-base-multilingual-uncased --train_file output/preprocessed/forBERT/en_train.txt --validation_file output/preprocessed/forBERT/en_val.txt --output_dir output/models/ --line_by_line --do_train --do_eval
```