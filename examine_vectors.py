import torch
from transformers import *
import pdb
import operator
from collections import OrderedDict
import sys

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


PATH='bert-base-cased'
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(PATH,do_lower_case=False)
model = BertForMaskedLM.from_pretrained(PATH)
model.eval()

def get_sent():
    print("Enter sentence:")
    sent = input()
    if (not sent.endswith(".")):
        print("Appending period to do dummy masking")
        sent = sent + " ."
    return '[CLS] ' + sent + '[SEP]'

def print_tokens(tokenized_text):
    dstr = ""
    for i in range(len(tokenized_text)):
        dstr += "   " +  str(i) + ":"+tokenized_text[i]
    print(dstr)
    print()

def get_pos():
    while True:
        masked_index = 0
        try:
                masked_index = int(input())
                return masked_index
        except:
            print("Enter valid number: (0 to quit)")
            masked_index = int(input())
            if (masked_index == 0):
                print("Quitting")
                sys.exit()
            return masked_index


while (True):
    text = get_sent()

    tokenized_text = tokenizer.tokenize(text)
    print_tokens(tokenized_text)
    #pdb.set_trace()
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    masked_index = len(tokenized_text) - 2
    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens[masked_index] = 103
    results_dict = {}

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    while True:
        print_tokens(tokenized_text)
        print("Enter any term position neighbor:")
        masked_index = get_pos()
        results_dict = {}
        for i in range(len(predictions[0][0,masked_index])):
            tok = tokenizer.convert_ids_to_tokens([i])[0]
            results_dict[tok] = float(predictions[0][0,masked_index][i].tolist())

        k = 0
        hist_d = {}
        sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
        first = True
        max_val = 0
        for i in sorted_d:
            if (first):
                max_val = sorted_d[i]
                first = False
            val = round(float(sorted_d[i])/max_val,1)
            if (val in hist_d):
                hist_d[val] += 1
            else:
                hist_d[val] = 1
            k += 1
            if (k <= 20):
                print(i,sorted_d[i])
        fp = open("top_k.txt","w")
        hist_d_sorted = OrderedDict(sorted(hist_d.items(), key=lambda kv: kv[0], reverse=False))
        for i in hist_d_sorted:
            fp.write(str(i) + " " + str(hist_d_sorted[i]) + "\n")
        fp.close()
