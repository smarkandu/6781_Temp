import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings

from data_collection import get_data2

warnings.filterwarnings('ignore')

full_vocab, human_vocab, chatgpt_vocab, df_train, df_test = get_data2()

# Combine datasets into a single DataFrame
df_train.columns = [0, 1]
print(df_train.head())

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

#tokenized = df_train[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized = df_train[0].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True))
#tokenized = df_train[0].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, padding='max_length', max_length=512))


# Find the maximum length
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

# Pad the values for those less than the maximum length
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print(tokenized.values[0])
print(np.array(padded).shape)

# Add the masking
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# Get Labels
features = last_hidden_states[0][:,0,:].numpy()
labels = df_train[1]

# Make a Train / Test Split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

score = lr_clf.score(test_features, test_labels)
print('Score', score)