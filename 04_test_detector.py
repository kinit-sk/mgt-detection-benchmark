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
from sklearn.metrics import classification_report
from transformers import pipeline
import numpy as np
import torch
import gc
import nvidia_smi, psutil, shutil
import time
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def report_gpu():
  nvidia_smi.nvmlInit()
  handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
  info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
  print("GPU [GB]:", f'{info.used/1024/1024/1024:.2f}', "/", f'{info.total/1024/1024/1024:.1f}')
  nvidia_smi.nvmlShutdown()
  print('RAM [GB]:', f'{psutil.virtual_memory()[3]/1024/1024/1024:.2f}', "/", f'{psutil.virtual_memory()[0]/1024/1024/1024:.1f}')

start = time.time()
classifier = pipeline("text-classification", model=output_model, device=device, torch_dtype=torch.float16)
end = time.time()
print(f"{output_model.split('/')[-1]} loading took {(end - start)/60} min")
print(f"{output_model.split('/')[-1]} memory footprint {classifier.model.get_memory_footprint()/1024/1024/1024} GB")
report_gpu()

def predict(df):
  preds = ['unknown'] * len(df)
  scores = [0] * len(df)
  for index, row in tqdm(df.iterrows(), total=len(df)):
    tokenizer_kwargs = {'truncation':True,'max_length':512}
    pred = classifier(row['text'], **tokenizer_kwargs)
    preds[index] = pred[0]['label']
    scores[index] = pred[0]['score']
  return preds, scores
    
test = pd.read_csv(DATAPATH)
test = test[test.split == "test"].reset_index(drop=True)
test['label'] = ["human" if "human" in x else "machine" for x in test.multi_label]

start = time.time()
preds = predict(test)
test['predictions'] = preds[0]
test['prediction_probs'] = preds[1]
end = time.time()
print(f"{output_model.split('/')[-1]} testing took {(end - start)/60} min")
print(f"{output_model.split('/')[-1]} memory footprint {classifier.model.get_memory_footprint()/1024/1024/1024} GB")
report_gpu()

test.to_csv(f"{DATAPATH.replace('multitude.csv','results/finetuned/')}{output_model.split('/')[-1]}.csv.gz", compression='gzip', index=False)
print(classification_report(test['label'], test['predictions'], digits=4))
