---
title: Predict Earnings Call Sentiment using Current State of the Art Models (BERT, XLNet)
date: 2020-04-15
#tags: [Python, Earnings Call, NLP, BERT, XLNet, Training on GPU]
header:
  image: "/images/stock.jpg"
mathjax: "true"
---

# Part 1: Out-of-Sample Training: Training BERT and XLNet using Amazon Review Dataset on Virtual Machine GPU

After getting thousands of earnings call transcripts using the methods in the previous post, this section will go through how to predict sentiment for these transcripts using the current state of the art NLP models, BERT and XLNet.   


## Part 1a: BERT

This [git repo](https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/08.sentiment-analysis-with-bert.ipynb) was extremely useful for learning about BERT and how to implement it.

To summarize, BERT stands for Bidirectional Encoder Representations from Transformers and was developed by the Google Brain research team in 2018.  It is a lot more effective compared to CNNs and LSTMs in sentiment analysis since "Bi-directional" means it uses surrounding words (instead of reading in a single direction) and the "Transformers" make use of the attention mechanism to learn contextual references.  The words are encoded in a pre-trained embedding layer which gets fed into a neural network.  

I will also be training my BERT and XLNet model (in the next section) on a virtual machine GPU since these models are too big to run locally on my PC.  I am using [Paperspace Gradient](https://www.paperspace.com) which provides an environment for Jupyter notebooks to run on their cloud GPUs.  You can sign up for a free account which will give you access to a GPU for 6 hours after which the connection will automatically close so you have to make sure your model finishes training in that time or you can split up your training set and run multiple 6 hour sessions.


Depending on the container you choose in Paperspace Gradient, these are the main packages that you need to install:

```cmd
pip install torch
pip install transformers
pip install sklearn
```

Import packages:

```python 
import os, sys
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import watermark
from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
```


Let's load in our Amazon Review Polarity dataset, this dataset can be easily found online and it's massive so I'm only going to take a slice of it to train on:

```python
# load in data from csv
train_path = os.path.join(paths['train'], 'amazon_review_polarity_csv')
train_df = pd.read_csv(os.path.join(train_path, 'train_0.csv'), encoding = "ISO-8859-1", header=None)
train_df.columns = ['sentiment', 'review_title', 'review'] 
train_df['sentiment'] = train_df['sentiment'].map({1: 0, 2: 1})

class_names = [1, 0]
```

Now load in the tokenizer for the pre-trained Bert model, there are different versions available (base or large, cased or uncased). 
We also need to check the length of the tokenized reviews so we can select a max length that wouldn't cut off too much information.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

token_lens = []
for txt in train_df.review:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 500]);
plt.xlabel('Token count')
```


Here's a histogram of the token's length:

[![](/assets/images/predict_earning_call_transcripts/bert_histogram.JPG)](/assets/images/predict_earning_call_transcripts/bert_histogram.JPG)


Now we will set the max length, create the the dataset class and split our training data into train, validation, and test set.

```python
MAX_LEN = 150 
RANDOM_SEED = 1
BATCH_SIZE = 16

class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


df_train, df_test = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
```


Now we load and encode our split training dataset and load in the pre-trained Bert model:

```python
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
    reviews=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
      )

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'],
  attention_mask=encoding['attention_mask'] 
  )
```


Now we define our classifier, optimizer, loss function, accuracy function, and training and validation functions:

```python
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = SentimentClassifier(len(class_names))
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        #input_ids = d["input_ids"]
        #attention_mask = d["attention_mask"]
        #targets = d["targets"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)
          #input_ids = d["input_ids"]
          #attention_mask = d["attention_mask"]
          #targets = d["targets"]
          outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
          _, preds = torch.max(outputs, dim=1)
          loss = loss_fn(outputs, targets)
          correct_predictions += torch.sum(preds == targets)
          losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def compute_accuracy(predictions,labels):
    prob = predictions
    t = np.ones(len(predictions)) * 0.5   # threshold
    predictions_to_binary = (prob > t) * 1
    equality = predictions_to_binary==labels
    accuracy = equality.mean()
    return accuracy
```


Here is the main training and validation loop:

```python
t0 = time.time()

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')
      
    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(df_val))
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), r'/notebooks/best_model_state_maxlen'+str(MAX_LEN)+'.bin')
        best_accuracy = val_acc
        print(val_acc)

t1 = time.time()
print(t1-t0)
```


If we are satisfied with the performance on the validation set, we can use this model to predict on the sentiment on the earnings call transcripts. Here's what the test function looks like if we were to perform it on our existing Amazon Review dataset test split.

```python
def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)
```


I got a validation accuracy of about 93.2% on the Amazon Review Polarity dataset using BERT but decided to predict the earnings call sentiment using XLNet because that validation accuracy was even better. We will walk through that in the next post. 



