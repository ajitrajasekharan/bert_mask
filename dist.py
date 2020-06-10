import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from transformers import *
import sys


try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def read_embeddings(embeds_file):
    with open(embeds_file) as fp: 
        embeds_dict = json.loads(fp.read())
    return embeds_dict


def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 1 
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1): 
                terms_dict[term] = count
                count += 1
            else:
                print("skipping token:" + str(count) + " " + term)
    print("count of tokens:", len(terms_dict))
    return terms_dict




class BertEmbeds:
    def __init__(self, model_path,do_lower, terms_file,embeds_file,cache_embeds):
        do_lower = True if do_lower == 1 else False
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        #self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        #model = RobertaForMaskedLM.from_pretrained(model_path)
        #print(self.tokenizer.vocab_size)
        #self.dump_vocab()
        #sys.exit(-1)
        self.terms_dict = read_terms(terms_file)
        self.embeddings = read_embeddings(embeds_file)
        self.cache = cache_embeds
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.dist_threshold_cache = {}

    def dump_vocab(self):
        #pdb.set_trace()
        size = self.tokenizer.vocab_size
        for i in range(size):
            names = self.tokenizer.convert_ids_to_tokens([i])
            print(names[0])

    def gen_dist_for_vocabs(self):
        count = 1
        picked_count = 0
        cum_dict = OrderedDict()
        cum_dict_count = OrderedDict()
        for key in self.terms_dict:
            if (count <= 106 or str(key).startswith('#') or str(key).startswith('[')): #Words selector. skiping all unused and special tokens
            #if (count <= 106 or not (str(key).startswith('#'))): #subwords selector. skipping all unused and special tokens
                count += 1
                continue
            #print(":",key)
            picked_count += 1
            sorted_d = get_distribution_for_term(self,key,False)
            for k in sorted_d:
                val = round(float(k),1)
                #print(str(val)+","+str(sorted_d[k]))
                if (val in cum_dict):
                    cum_dict[val] += sorted_d[k]
                    cum_dict_count[val] += 1
                else:
                    cum_dict[val] = sorted_d[k]
                    cum_dict_count[val] = 1
        for k in cum_dict:
            cum_dict[k] = float(cum_dict[k])/cum_dict_count[k]
        final_sorted_d = OrderedDict(sorted(cum_dict.items(), key=lambda kv: kv[0], reverse=False))
        print("Total picked:",picked_count)
        with open("cum_dist.txt","w") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            for k in final_sorted_d:
                print(k,final_sorted_d[k])
                p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                fp.write(p_str)




    def get_embedding(self,text,tokenize=True):
        if (self.cache and text in self.embeds_cache):
            return self.embeds_cache[text]
        if (tokenize):
            tokenized_text = self.tokenizer.tokenize(text)
        else:
            tokenized_text = text.split()
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        #print(text,indexed_tokens)
        vec =  self.get_vector(indexed_tokens)
        if (self.cache):
                self.embeds_cache[text] = vec
        return vec


    def get_vector(self,indexed_tokens):
        vec = None
        if (len(indexed_tokens) == 0):
            return vec
        for i in range(len(indexed_tokens)):
            term_vec = self.embeddings[indexed_tokens[i]]
            if (vec is None):
                vec = np.zeros(len(term_vec))
            vec += term_vec
        sq_sum = 0
        for i in range(len(vec)):
            sq_sum += vec[i]*vec[i]
        sq_sum = math.sqrt(sq_sum)
        for i in range(len(vec)):
            vec[i] = vec[i]/sq_sum
        #sq_sum = 0
        #for i in range(len(vec)):
        #    sq_sum += vec[i]*vec[i]
        return vec

    def calc_inner_prod(self,text1,text2,tokenize):
        if (self.cache and text1 in self.cosine_cache and text2 in self.cosine_cache[text1]):
            return self.cosine_cache[text1][text2]
        vec1 = self.get_embedding(text1,tokenize)
        vec2 = self.get_embedding(text2,tokenize)
        #pdb.set_trace()
        if (vec1 is None or vec2 is None):
            #print("Warning: at least one of the vectors is None for terms",text1,text2)
            return 0
        val = np.inner(vec1,vec2)
        if (self.cache):
            if (text1 not in self.cosine_cache):
                self.cosine_cache[text1] = {}
            self.cosine_cache[text1][text2] = val
        return val


def get_word():
    while (True):
        print("Enter a word : q to quit")
        sent = input()
        #print(sent)
        if (sent == "q"):
            print("Exitting")
            sys.exit(1)
        if (len(sent) > 0):
            break
    return sent

debug_fp = None
hack_check = False
def get_distribution_for_term(b_embeds,term1,tokenize):
    global debug_fp
    if (term1 in b_embeds.dist_threshold_cache):
        return b_embeds.dist_threshold_cache[term1]
    terms_count = b_embeds.terms_dict
    dist_dict = {}
    val_dict = {}
    if (hack_check and debug_fp is None):
        debug_fp = open("debug.txt","w")
    for k in b_embeds.terms_dict:
        term2 = k.strip("\n")
        val = b_embeds.calc_inner_prod(term1,term2,tokenize)
        #if (hack_check and val >= .8 and term1 != term2):
        if (hack_check and val >= .6 and val < .8 and term1 != term2):
            print(term1,term2)
            str_val = term1 + " " + term2 + "\n"
            debug_fp.write(str_val)
            debug_fp.flush()

        val = round(val,2)
        if (val in dist_dict):
            dist_dict[val] += 1
        else:
            dist_dict[val] = 1
    sorted_d = OrderedDict(sorted(dist_dict.items(), key=lambda kv: kv[0], reverse=False))
    b_embeds.dist_threshold_cache[term1] = sorted_d
    return sorted_d

def print_terms_above_threshold(b_embeds,term1,threshold,tokenize):
    final_dict = {}
    fp = open("above_t.txt","w")
    for k in b_embeds.terms_dict:
        term2 = k.strip("\n")
        val = b_embeds.calc_inner_prod(term1,term2,tokenize)
        val = round(val,2)
        if (val > threshold):
            final_dict[term2] = val
    sorted_d = OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True))
    for k in sorted_d:
            print(k," ",sorted_d[k])
            fp.write(str(k) + " " + str(sorted_d[k]) + "\n")
    fp.close()



def pick_threshold():
    while (True):
        print("Enter threshold to see words above threshold: q to quit")
        sent = input()
        if (sent == "q"):
            print("Exitting")
            sys.exit(1)
        try:
            thres = float(sent)
            return thres
        except:
            print("Invalid input. Retry")



def main():
    if (len(sys.argv) != 6):
        print("Usage: <Bert model path - to load tokenizer> do_lower_case[1/0] <vocab file> <vector file> <tokenize text>1/0")
    else:
        tokenize = True if int(sys.argv[5]) == 1 else False
        print("Tokenize is set to :",tokenize)
        b_embeds =BertEmbeds(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],True) #True - for cache embeds
        #b_embeds.gen_dist_for_vocabs() #uncomment this line to get distributions
        #return
        while (True):
            word = get_word()
            if (tokenize):
                tokenized_text = b_embeds.tokenizer.tokenize(word)
                print("Tokenized text:", tokenized_text)
            #pdb.set_trace()
            sorted_d = get_distribution_for_term(b_embeds,word,tokenize)
            for k in sorted_d:
                print(str(k)+","+str(sorted_d[k]))
            if (tokenize):
                print("Tokenized text:", tokenized_text)
            threshold = pick_threshold()
            print_terms_above_threshold(b_embeds,word,threshold,tokenize)




if __name__ == '__main__':
    main()
