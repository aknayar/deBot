#This is to turn the code into text embeddings using gensim
#Might work

from collections import defaultdict
import numpy as np
import pickle
import gensim

from gensim.models.keyedvectors import KeyedVectors

#Instead of using Justin's break, one can use the function that was created

"""
from gensim.models import KeyedVectors

 path = get_tmpfile("wordvectors.kv")

 model.wv.save(path)
 wv = KeyedVectors.load("model.wv", mmap='r')
 vector = wv['computer']  # numpy vector of a word

Will deivide the text into sentences
return [sent for sent in document.sents]
"""
#an array of sentences
#an array of embeddings

path = r".\glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)


def to_embeddings(arr_statements):
    #return a tensor - retrieve the underlying NumPy array.
    #my_vectors = model.encode(data).data

    """
    :param
    text:string of our captions
    :return
    vectors: a (50,)np.ndarray that represents out caption in semantic space
    """

    vectors = np.zeros((len(arr_statements),))
    for sentence in arr_statements:

        if sentence not in glove:
            pass
        else:
            sentence_vector = glove[sentence]


            vectors += sentence_vector

    return vectors.reshape((1, len(arr_statements)))