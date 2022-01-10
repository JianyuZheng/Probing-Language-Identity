# coding=gbk#coding:utf-8
import argparse
import torch
from transformers import BertTokenizer, BertModel, XLMTokenizer, XLMModel,XLMRobertaModel, XLMRobertaTokenizer
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
    for layer in range(1,13):
        b = []
        for line in lines:
            if args.model == 'bert':
                sent = "[CLS] " + line.strip() + " [SEP]"
            else:
                sent = line.strip()
            tokenized_text = tokenizer.tokenize(sent)                                        #�Ծ�����token������
            if len(tokenized_text) > args.max_len:                                           #�س�����
                tokenized_text = tokenized_text[:args.max_len]
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)                 #ת��������id
            segments_ids = [1] * len(tokenized_text)                                         #ȷ���ָ��

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)  #�õ����ӵ���������ʾ
                encoded_layers = outputs[-1][layer]
                token_vecs = encoded_layers[0]
                #���ݲ�ͬģ�ͣ����ò�ͬ����������ʾ����
                if args.model == 'bert':
                    sentence_embedding = token_vecs[0]                                       #ȡ[SEP]��������ʾ
                else:
                    sentence_embedding = torch.mean(token_vecs, dim=0)                       #��ֵ������
                b.append(sentence_embedding.detach().numpy())

        #�洢���
        path = './'+args.model+'/layer '+str(layer)
        if not os.path.exists(path):
            os.makedirs(path)
        newfile = path +'/'+file.split('.')[0]+'_'+args.model+'_embedding.dat'
        with open(newfile, 'wb') as f:
            pickle.dump(b, f)

#������
def main(args):
    #������䣬���ڷ����жϾ����ǲ�����һ��Ԥѵ��������ģ��
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')                 #�����token�Ĺ���
        model = BertModel.from_pretrained("bert-base-multilingual-cased")                         #����Ԥѵ��ģ���ļ�
    elif args.model == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")                       #xlm-mlm-xnli15-102
        model = XLMModel.from_pretrained("xlm-mlm-tlm-xnli15-1024")
    elif args.model == 'xlm-R':
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    model.eval()                              #ģ������
    files = os.listdir(args.source_text)
    for file in files:                       #�����ļ����¸������ֵ��ļ�
        run(file, tokenizer, model)


if __name__ == "__main__":
    #��Ҫ�Ĳ���
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xlm', choices=['bert', 'xlm', 'xlm-R'])    #ѡ����һ��Ԥѵ��������ģ��
    parser.add_argument('--max_len', type=int, default=512)                                      #������󳤶�
    parser.add_argument('--source_text', type=str, default='./multilingual text')                #���ϵ��ļ���

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                #�豸(gpu or cpu)

    print(args)
    main(args)
