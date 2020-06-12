---
title: Predict Earnings Call Sentiment using Current State of the Art Models (BERT, XLNet) (Cont'd)
date: 2020-04-30
#tags: [Python, Earnings Call, NLP, BERT, XLNet, Training on GPU]
header:
  image: "/images/stock.jpg"
mathjax: "true"
---

# Part 1: Out-of-Sample Training: Training BERT and XLNet using Amazon Review Dataset on Virtual Machine GPU (Cont'd)


## Part 1b: XLNet

XLNet was developed by Google and CMU researchers in 2019. It is similar to the BERT model but has key differences that allows it to outperform BERT on numerous NLP tasks.  One of the main differences is that unlike BERT, it does not use a mask token so it does not make independence assumptions between masked tokens.  XLNet is also a deeper multilayered network than BERT so it can capture dependencies in longer sequences but this means it also uses more memory and takes longer to train. 

This [link](https://mccormickml.com/2019/09/19/XLNet-fine-tuning) was extremely useful in learning about how to implement the pre-trained XLNet model.

Similar to the previous post on BERT, I will also be using the cloud GPU on [Paperspace Gradient](https://www.paperspace.com) to train the XLNet model. Let's jump into the code, I've set this up very similar to the code from the BERT post.


Depending on the container you choose in Paperspace Gradient, these are the main packages that you need to install:

```cmd
pip install torch
pip install tensorflow
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
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from transformers import *
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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

Now load in the tokenizer for the pre-trained XLNet model, there are also different versions available like the BERT model (base or large, cased or uncased). 
We also need to check the length of the tokenized reviews so we can select a max length that wouldn't cut off too much information.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)

sentences = train_df.review.values
sentences = [sentence + " [SEP] [CLS]" for sentence in sentences]
labels = train_df.sentiment.values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

token_lens = []
for txt in train_df.review:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 500]);
plt.xlabel('Token count')
```


Here's a histogram of the token's length:

[![](/assets/images/predict_earning_call_transcripts_p2/xlnet_histogram.JPG)](/assets/images/predict_earning_call_transcripts_p2/xlnet_histogram.JPG)


Now we will set the max length, use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary, and pad the input tokens:

```python
MAX_LEN = 300 

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
```


Now we will create attention mask of 1s for each token followed by 0s for padding:

```python
attention_masks = []
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)
```


Now we will split our training data into train and validation set and convert the input data into torch tensors.

```python
RANDOM_SEED = 1
BATCH_SIZE = 8

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=RANDOM_SEED, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=RANDOM_SEED, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
```


Now we load our input data into in iterator (saves memory usage compared to looping), load in the pre-trained XLNet model and set the variables used in our training loop:

```python
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
model.cuda()

```


Now we define our optimizer and accuracy function:

```python

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
```


Here is the main training and validation loop:

```python
t0 = time.time()
best_accuracy = 0

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs 
epochs = 4

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):

    # Training

    # Set our model to training mode 
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        if tmp_eval_accuracy > best_accuracy:
            torch.save(model.state_dict(),
                       r'/notebooks/xlnet_model_maxlen'+ str(MAX_LEN)+'.bin')
            best_accuracy = tmp_eval_accuracy
            print(tmp_eval_accuracy)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

t1 = time.time()
print(t1 - t0)
```

After fine tuning this pre-trained XLNet model using the Amazon Review Polarity dataset, I got a validation accuracy of 94.7% which is quite a bit better than BERT. So I will save down this model and use it to predict the sentiment on the earnings call transcripts. 

At the end of the "Getting Earnings Call Transcripts" post, I combined the preprocessed transcripts into one csv file and this is what is going to be fed to the saved down XLNet model.

We just need to upload this csv to the virtual machine Paperspace Gradient, to upload files just click on the highlighted button:

[![](/assets/images/predict_earning_call_transcripts_p2/upload_paperspace.JPG)](/assets/images/predict_earning_call_transcripts_p2/upload_paperspace.JPG)


Then I can run these functions below which will load in my saved down XLNet model called "xlnet_model_state_maxlen300_train_0_acc94.7_4epoch.bin" and load in the test data that I just uploaded called "combined_trans.csv":

```python

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


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(reviews=df.review.to_numpy(), targets=df.sentiment.to_numpy(), tokenizer=tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, 
                      num_workers=4)


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

            #input_ids = d["input_ids"]
            #attention_mask = d["attention_mask"]
            #targets = d["targets"]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs[0], dim=1)
            probs = F.softmax(outputs[0], dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)

          outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
          _, preds = torch.max(outputs[0], dim=1)
          loss = loss_fn(outputs[0], targets)
          correct_predictions += torch.sum(preds == targets)
          losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def xlnet_predict_trans_sentiment(i):
    t0 = time.time()
    MAX_LEN = 300
    BATCH_SIZE = 8
    PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
    class_names = [1, 0]
    tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)
    path = os.path.join('/notebooks')
    
    # load in saved down model
    model = XLNetForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=len(class_names)).to(device)
    model.load_state_dict(torch.load(os.path.join(path,'xlnet_model_state_maxlen300_train_0_acc94.7_4epoch.bin' )))
    
    df = pd.read_csv(os.path.join(path, 'combined_trans.csv'), encoding='utf-8')
    test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)
    df = df.rename(columns={'review': 'content'})
    df['pred'] = y_pred
    df['pred_probs'] = y_pred_probs
    df.to_csv(os.path.join(path, 'combined_trans_predictions.csv'), index=False)
    t1 = time.time()
    print(t1-t0)
    print('combined predictions csv saved: ' + str(os.path.join(path, 'combined_trans_predictions.csv')))


xlnet_predict_trans_sentiment()
```

The predicted sentiment for all the call transcripts is saved in this file: 'combined_trans_predictions.csv'. We can easily download this from the VM on Paper Gradient to our local machine and do further analysis on this later.

Here's what predicted call transcript looks like:

[![](/assets/images/predict_earning_call_transcripts_p2/predicted.JPG)](/assets/images/predict_earning_call_transcripts_p2/predicted.JPG)

