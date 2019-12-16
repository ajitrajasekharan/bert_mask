import torch
from transformers import *
import pdb
import operator
from collections import OrderedDict
import sys
import traceback
import argparse


import logging

DEFAULT_MODEL_PATH='bert-base-cased'
DEFAULT_TO_LOWER=False
DEFAULT_TOP_K = 20
ACCRUE_THRESHOLD = 1

def init_model(model_path,to_lower):
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()
    return model,tokenizer

def get_sent():
    print("Enter sentence. Type q to quit")
    sent = input()
    if (sent == 'q'):
        return sent
    else:
        return '[CLS] ' + sent + '[SEP]'


def get_mask_index(limit):
    masked_index = 0
    while (True):
        try:
            print("Enter mask index value in range 0 -",limit-1)
            masked_index = int(input())
            if (masked_index < limit and masked_index >= 0):
                break
        except:
            print("Enter Numeric value:")
    return masked_index


def perform_task(model,tokenizer,top_k,accrue_threshold,text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    masked_index = 0

    for i in range(len(tokenized_text)):
        if (tokenized_text[i] == "entity"):
            masked_index = i
            break
    if (masked_index == 0):
        dstr = ""
        for i in range(len(tokenized_text)):
            dstr += "   " +  str(i) + ":"+tokenized_text[i]
        print(dstr)
        masked_index = get_mask_index(len(tokenized_text))

    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens[masked_index] = 103
    print(tokenized_text)
    print(masked_index)
    results_dict = {}

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
        for i in range(len(predictions[0][0,masked_index])):
            if (float(predictions[0][0,masked_index][i].tolist()) > accrue_threshold):
                tok = tokenizer.convert_ids_to_tokens([i])[0]
                results_dict[tok] = float(predictions[0][0,masked_index][i].tolist())

    k = 0
    sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
    for i in sorted_d:
        print(i,sorted_d[i])
        k += 1
        if (k > top_k):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting neighbors to a word in sentence using BERTMaskedLM. Neighbors are from BERT vocab (which includes subwords and full words). Type in a sentence and then choose a position to mask or type in a sentence with the word entity in the location to apply a mask ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-topk', action="store", dest="topk", default=DEFAULT_TOP_K,type=int,help='Number of neighbors to display')
    parser.add_argument('-tolower', action="store", dest="tolower", default=DEFAULT_TO_LOWER,help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-threshold', action="store", dest="threshold", default=ACCRUE_THRESHOLD,type=float,help='threshold of results to pick')

    results = parser.parse_args()
    try:
        model,tokenizer = init_model(results.model,results.tolower)
        while (True):
            text = get_sent()
            if (text == "q"):
                print("Quitting")
                break
            perform_task(model,tokenizer,results.topk,results.threshold,text)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
