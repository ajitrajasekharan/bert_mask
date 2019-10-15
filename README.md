# bert_mask

This is an example program illustrating BERTs masked language model. 
Given a sentence as input, we can specify any term (could be a subword of a word) to mask and examine its neighbors, where the neighbors are terms in BERT's vocab.
We can use this for a variety of tasks
* To fill in missing puncuations in a sentence. 
* To harvest phrases of a particular entity type (all phrases beloing to a particular entity type, are likely to share common neighbor terms in the top k neighbors in a sentence a term of that entity type occurs. 
* In general any task where the sentence context of a word/phrase taking would be useful. 

# Install steps
*Install pytorch first. This link (https://github.com/ajitrajasekharan/multi_gpu_test)   has installation instructions for pytorch
*Activate the environment if using conda

# Usage 
* python mask_word.py

# Sample outputs
A sentence "He went to prison _cell_ with his _cell_ phone to extract blood _cell_ samples from inmates" with the word cell having different senses. 

![Output of mask_word.py - 1 of 3](1.png) 

![Output of mask_word.py - 2 of 3](2.png) 

![Output of mask_word.py - 3 of 3](3.png) 

The neighbors for the word "cell" in the sentence above are different for the different contexts. Note all displayed neighbors are words in BERT vocab. This test was done using pretrained model - bert-base-cased


# License

MIT License
