# coding=gbk#coding:utf-8
#���ļ����mbert�Ծ��ӵ���������ʾ
import argparse
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig
import os
import numpy as np
import pickle


#run���������ڽ������Եľ��ӽ�����������ʾ
def run(file, tokenizer, model):
    #�����ı�
    with open(args.source_text+'/'+file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

    #��ģ�͵ĸ��������ѭ��
    b = []
    for line in lines:
        encoded_input = tokenizer(line.strip(), return_tensors='pt')
        outputs = model(**encoded_input)
        encoded_layers = outputs[0]
        token_vecs = encoded_layers[0]
        sentence_embedding = token_vecs[0]
    
        b.append(sentence_embedding.detach().numpy())    #�õ����ӵ���������ʾ

        #�洢���
        path = './'+args.model+'/layer '+str(12)
        if not os.path.exists(path):
            os.makedirs(path)
        newfile = path +'/'+file.split('.')[0]+'_'+args.model+'_embedding.dat'
        with open(newfile, 'wb') as f:
            pickle.dump(b, f)


#������
def main(args):
    configuration = BertConfig.from_pretrained('config.json')                       #��������
    model = BertModel(configuration)                                                #����ģ��
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')       #��token�Ĺ���

    model.eval()                               #ģ������
    files = os.listdir(args.source_text)
    for file in files:                         #�����ļ����¸������ֵ��ļ�
        run(file, tokenizer, model)


if __name__ == "__main__":
    #��Ҫ�Ĳ���
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert_random')                  #ѡ����һ��Ԥѵ��������ģ��
    parser.add_argument('--max_len', type=int, default=512)                          #������󳤶�
    parser.add_argument('--source_text', type=str, default='./multilingual text')    #���ϵ��ļ���

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    #�豸(gpu or cpu)

    print(args)
    main(args)
