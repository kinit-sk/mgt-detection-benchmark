DATAPATH = "./multitude.csv" #path of the dataset for training
MODELPATH = "./finetuned_models/" #path where fine-tuned model will be saved
CACHE = "./cache/" #change to your huggingface cache folder (where the pretrained models will be downloaded)

import sys

PRE_TRAINED_MODEL_NAME = sys.argv[1]
model_name = PRE_TRAINED_MODEL_NAME.split('/')[-1]
dataset = sys.argv[2] #'en', 'es', 'ru', 'all', 'en3'
generative_model = sys.argv[3] #'text-davinci-003', 'gpt-3.5-turbo', 'gpt-4', 'llama-65b', 'opt-66b', 'opt-iml-max-1.3b', 'all'
output_model = f'{MODELPATH}{model_name}-finetuned-{dataset}-{generative_model}'
balance = False
if balance:
    output_model = f'{MODELPATH}{model_name}-finetuned-{dataset}-{generative_model}-balanced'

import os
os.environ['HF_HOME'] = CACHE

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.optimization import Adafactor, AdafactorSchedule
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import torch
import gc
import nvidia_smi, psutil, shutil
import time

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

label_names = ["human", "machine"] #0, 1
id2label = {idx:label for idx, label in enumerate(label_names)}
label2id = {v:k for k,v in id2label.items()}

def map_labels(example):
  label_name = example["label"]
  return {"label": label2id[label_name], "label_name": label_name}
    
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, cache_dir=CACHE)

if tokenizer.pad_token is None:
  if tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  else:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

start = time.time()
num_labels = len(label_names)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, cache_dir=CACHE, num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))
try:
  model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
except:
  print("Warning: Exception occured while setting pad_token_id")
end = time.time()
print(f'{model_name} loading took {(end - start)/60} min')
print(f'{model_name} memory footprint {model.get_memory_footprint()/1024/1024/1024}')

train = pd.read_csv(DATAPATH)
train = train[train.split == "train"]

#language selection
if dataset == "en":
    train = train[train.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = RANDOM_SEED)).sample(frac=1., random_state = 0).reset_index(drop=True)
elif dataset == "en3":
    train = train[train.language == "en"].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif dataset == "es":
    train = train[train.language == "es"].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif dataset == "ru":
    train = train[train.language == "ru"].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif dataset == "all":
    train_en = train[train.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = RANDOM_SEED)).sample(frac=1., random_state = 0).reset_index(drop=True)
    train_es = train[train.language == "es"]
    train_ru = train[train.language == "ru"]
    train = pd.concat([train_en, train_es, train_ru], ignore_index=True, copy=False).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)

#machine-text generation model selection
if generative_model != "all":
    train = train[train.multi_label.str.contains("human") | train.multi_label.str.contains(generative_model)].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
    
train['label'] = ["human" if "human" in x else "machine" for x in train.multi_label]

if(balance):
  train = train.groupby(['label']).apply(lambda x: x.sample(train.label.value_counts().max(), replace=True, random_state = RANDOM_SEED)).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)

valid = train[-(len(train)//10):]
train = train[:-(len(train)//10)]

print(train.groupby('language')['multi_label'].value_counts())
print(train.label.value_counts())

train = Dataset.from_pandas(train, split='train')
valid = Dataset.from_pandas(valid, split='validation')
train = train.map(map_labels)
valid = valid.map(map_labels)

def tokenize_texts(examples):
  return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train.map(tokenize_texts, batched=True)
tokenized_valid = valid.map(tokenize_texts, batched=True)

batch_size = 16
gradient_accumulation_steps=4
num_train_epochs = 10
learning_rate=2e-4
metric_for_best_model = 'MacroF1'
logging_steps = len(tokenized_train) // (batch_size * num_train_epochs)
logging_steps = round(2000 / (batch_size * gradient_accumulation_steps)) #eval around each 2000 samples


if ("small" in model_name):
    #logging_steps //= 3
    logging_steps *= 5
    #learning_rate=2e-8
    metric_for_best_model = 'ACC'

use_fp16 = True
if "mdeberta" in model_name: use_fp16 = False

args = TrainingArguments(
    output_dir=output_model,
    evaluation_strategy = "steps",
    logging_steps = logging_steps, #50,
    save_strategy="steps",
    save_steps = logging_steps, #50,
    save_total_limit=5,
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",
    metric_for_best_model = metric_for_best_model,
    fp16=use_fp16 #mdeberta not working with fp16
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"ACC": accuracy_score(labels, predictions), "MacroF1": f1_score(labels, predictions, average='macro'), "MAE": mean_absolute_error(labels, predictions)}

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
class MyAdafactorSchedule(AdafactorSchedule):
    def get_lr(self):
        opt = self.optimizer
        if "step" in opt.state[opt.param_groups[0]["params"][0]]:
            lrs = [opt._get_lr(group, opt.state[p]) for group in opt.param_groups for p in group["params"]]
        else:
            lrs = [args.learning_rate] #just to prevent error in some models (mdeberta), return fixed value according to set TrainingArguments
        return lrs #[lrs]
lr_scheduler = MyAdafactorSchedule(optimizer)
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    optimizers=(optimizer, lr_scheduler)
)

start = time.time()
trainer.train()
end = time.time()
print(f'{model_name} memory footprint {model.get_memory_footprint()/1024/1024/1024}')
print(f'{model_name} fine-tuning took {(end - start)/60} min')

start = time.time()
shutil.rmtree(output_model, ignore_errors=True)
trainer.save_model()
end = time.time()
print(f'{output_model} saving took {(end - start)/60} min')