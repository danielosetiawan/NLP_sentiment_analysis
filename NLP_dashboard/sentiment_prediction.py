import transformers
import torch
import math
import pandas as pd
import numpy as np
import emoji
import re
from transformers import (
    RobertaForSequenceClassification, RobertaTokenizer, BertForSequenceClassification, 
    BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AdamW
)
import random
import time


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


tokenizer = RobertaTokenizer.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
model = RobertaForSequenceClassification.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Sentiment(sent,model=model,tokenizer=tokenizer):
    encoded_dict = tokenizer.encode_plus(
                      sent, 
                      add_special_tokens = True,
                      truncation=True,
                      max_length = 64,
                      padding='max_length',
                      return_attention_mask = True,
                      return_tensors = 'pt')

    input_id = torch.LongTensor(encoded_dict['input_ids']).to(device)
    attention_mask = torch.LongTensor(encoded_dict['attention_mask']).to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(input_id, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs[0]
    index = logits.argmax()
    return index,logits


def process_text(texts):
    # lowercase
    # message = message.lower() # RoBERTa tokenizer is uncased
    # remove URLs
    texts = re.sub(r'https?://\S+', "", texts)
    texts = re.sub(r'www.\S+', "", texts)
    # remove '
    texts = texts.replace('&#39;', "'")
    # remove symbol names
    texts = re.sub(r'(\#)(\S+)', r'hashtag_\2', texts)
    texts = re.sub(r'(\$)([A-Za-z]+)', r'cashtag_\2', texts)
    # remove usernames
    texts = re.sub(r'(\@)(\S+)', r'mention_\2', texts)
    # demojize
    texts = emoji.demojize(texts, delimiters=("", " "))

    return texts.strip()

def checkSenti(sent,return_logits=True):
    labels = ['Bearish','Bullish']
    sent_processed = process_text(sent)
    index,logits = Sentiment(sent_processed)
    if return_logits:
        logit0 = math.exp(logits[0][0])
        logit1 = math.exp(logits[0][1])
        logits = [logit0/(logit0+logit1),logit1/(logit0+logit1)]
        return [labels[index], max(logits)]
    return labels[index]

