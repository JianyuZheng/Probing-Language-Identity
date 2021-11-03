import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig
import numpy as np
import os
import re
import random
import pickle
import time




tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")
model.eval()


start = time.time()
with open('./multilingual sentences/Catalan.txt', 'r', encoding='utf-8') as fp:
    lines = fp.readlines()
    lines = [line.strip() for line in lines]
    #lines = [line.strip().split(None, 1)[-1] for line in lines]
    #lines = [line.strip().split(None, 2)[-1] for line in lines]  #for Greek (Modern).txt
    #random.shuffle(lines)
    #lines = lines[:10000]
  
b = []
for line in lines:
    sent = "[CLS] " + line.strip() + " [SEP]"

    tokenized_text = tokenizer.tokenize(sent)
    #print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    
    '''
    with torch.no_grad():  #只输出最终层
        outputs = model(tokens_tensor, segments_tensors)
        encoded_layers = outputs[0]
        print(encoded_layers.size())
        token_vecs = encoded_layers[0]
        sentence_embedding = token_vecs[0]
        #print(sentence_embedding.size())
        #b = sentence_embedding.numpy().tolist()
        b.append(sentence_embedding.numpy())
    '''
    
    with torch.no_grad(): 
        outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)
        encoded_layers = outputs[-1][1]   
        token_vecs = encoded_layers[0]
        sentence_embedding = token_vecs[0]
        b.append(sentence_embedding.numpy())
    
    
    

with open('Catalan_bert_embedding.dat', 'wb') as f:
    pickle.dump(b, f)

print(time.time()-start)



'''

with torch.no_grad():
    #outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)
    outputs = model(tokens_tensor, segments_tensors)
    encoded_layers = outputs[0]
    token_vecs = encoded_layers[0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    #print(sentence_embedding.size())
    b = sentence_embedding.numpy().tolist()
'''








