# bert_mask

This is an example program illustrating BERTs masked language model. 
Given a sentence as input, we can specify any term (could be a subword of a word) to mask and examine its neighbors, where the neighbors are terms in BERT's vocab.
We can use this for a variety of tasks
* To fill in missing puncuations in a sentence. 
* To harvest phrases of a particular entity type (all phrases beloing to a particular entity type, are likely to share common neighbor terms in the top k neighbors in a sentence a term of that entity type occurs. 

# Install steps
*Install pytorch first. This link goes through the details
*Activate the environment if using conda

# Usage 
* python mask_word.py

![Output of mask_word.py](1.png) (2.png) (3.png)

The neighbors for the word "cell" in the sentence above are different for the different contexts. Note all neighbors are words in BERT vocab. This test was done using pretrained model - bert-base-cased


# License

MIT License
