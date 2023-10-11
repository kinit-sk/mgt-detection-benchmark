CACHE = "./cache/"
DATAPATH = "./"
model_name = "ai-forever/mGPT"

import transformers
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE, load_in_8bit=True)
base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE)
base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
base_model.to(device)

# get average entropy of each token in the text
def get_entropy(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()
def entropy_criterion(text): return get_entropy(
        text, base_model, base_tokenizer, device)

criterion_fn = entropy_criterion

CAST_DICT = {'text': 'string', 'language': 'string', 'label': 'string', 'length': 'int16', 'source': 'string', 'domain': 'string', 'topic': 'string', 'split': 'string', 'multi_label': 'string', 'predictions': 'string', 'prediction_probs': np.float16}

multitude = pd.read_csv(DATAPATH + 'multitude.csv', dtype = CAST_DICT)

multitude_train = multitude[multitude.split == 'train'].reset_index(drop=True)
train_text = multitude_train['text']
train_label = multitude_train['label']

train_criterion = [criterion_fn(train_text[idx]) for idx in range(len(train_text))]
x_train = np.array(train_criterion)
x_train = np.expand_dims(x_train, axis=-1)
y_train = train_label

x_valid = x_train[-(len(x_train)//10):]
y_valid = y_train[-(len(y_train)//10):]
x_train = x_train[:-(len(x_train)//10)]
y_train = y_train[:-(len(y_train)//10)]

multitude_test = multitude[multitude.split == 'test'].reset_index(drop=True)
test_text = multitude_test['text']
test_label = multitude_test['label']
 
test_criterion = [criterion_fn(test_text[idx]) for idx in range(len(test_text))]
x_test = np.array(test_criterion)
x_test = np.expand_dims(x_test, axis=-1)
y_test = test_label

#RF hyperparameter tuning & classifier training
n_estimators = [10, 50, 100, 150, 300] # number of trees in the random forest
criterion = ['gini', 'entropy'] # function to measure the quality of a split
max_features = ['sqrt', 'log2', None] # number of features in consideration at every split
max_depth = [None, 10, 100] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 4, 6] # minimum sample number to split a node
min_samples_leaf = [1, 3] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,
                'criterion': criterion,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

clf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs = 2)
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 1000, cv = 5, verbose=2, random_state=RANDOM_SEED, n_jobs = 2)
clf_random.fit(np.nan_to_num(x_train), y_train)

print(clf_random.best_params_)

clf = RandomForestClassifier(**clf_random.best_params_, random_state=RANDOM_SEED, n_jobs = 2)

clf.fit(np.nan_to_num(x_train), y_train)

preds = clf.predict(np.nan_to_num(x_valid))
print(classification_report(y_valid, preds, digits=4, output_dict=False))

preds = clf.predict(np.nan_to_num(x_test))
print(classification_report(y_test, preds, digits=4, output_dict=False))

pd.DataFrame({'Predictions': preds}).to_csv(DATAPATH + 'results/statistical/predictions_entropy_RF-tuned.csv', index=False)
