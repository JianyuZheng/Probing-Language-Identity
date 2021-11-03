import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import numpy as np
import os
import re
import random
import pickle
import time

start = time.time()


tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large") #xlm-mlm-xnli15-102
model = XLMRobertaModel.from_pretrained("xlm-roberta-large")
model.eval()


with open('./multilingual sentences/Vietnamese.txt', 'r', encoding='utf-8') as fp:
    lines = fp.readlines()
    lines = [line.strip() for line in lines]
    #lines = [line.strip().split(None, 1)[-1] for line in lines]
    #lines = [line.strip().split(None, 2)[-1] for line in lines]  #for Greek (Modern).txt
    #random.shuffle(lines)
    #lines = lines[:10000]

 

b = []
for line in lines:
    sent = line.strip()
    
    tokenized_text = tokenizer.tokenize(sent)
    if len(tokenized_text)>512:
        tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

   
    with torch.no_grad():  
        outputs = model(tokens_tensor, segments_tensors)
        encoded_layers = outputs[0]
        token_vecs = encoded_layers[0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        b.append(sentence_embedding.numpy())
    '''
    with torch.no_grad():  
        outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)
        encoded_layers = outputs[-1][12]   
        token_vecs = encoded_layers[0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        b.append(sentence_embedding.numpy())
   '''


with open('Vietnamese_embedding.dat', 'wb') as f:
    pickle.dump(b, f)

print(time.time()-start)






