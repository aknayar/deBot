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
    Take in M things (1, to a (1,50) Then put them together
    :return
    vectors: a (50,)np.ndarray that represents out caption in semantic space
    """
    m = arr_statements.shape()[0]
    main_vector = np.zeros(m,)
    for sentence in arr_statements:
        vectors = np.zeros((50,))

        if sentence not in glove:
            pass
        else:
            word_vector = glove[sentence]

            vectors += word_vector

        main_vector = np.append(main_vector, vectors)




    return vectors.reshape((m, 50))