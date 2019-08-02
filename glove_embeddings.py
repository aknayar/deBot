import re
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


# Initializes glove, which contains the embeddings for each word.
glove = KeyedVectors.load_word2vec_format("glove.6B.50d.txt.w2v", binary=False)


def embed_sentence(sentences):
    """
    Takes a list of sentences and embeds each word in every sentence using glove.

    Parameters
    ----------
    sentences: List
        An list of sentences to be embedded

    Returns
    -------
    numpy.ndarray, shape = (len(sentences), maximum length of a single sentence, 50)
        A numpy array containing each of the word embeddings for each sentence.
    """
    ret = []
    m = 0
    for i in sentences:
        i = i.lower().replace("[ref]", "")
        i = "".join(c for c in i if c.isdigit() or c.isalpha() or c == " ")
        i = re.sub(r" \W+", " ", i)
        i = i.split()
        if len(i) > m:
            m = len(i)
        row = []
        for word in i:
            try:
                row.append(glove[word])
            except:
                continue
        ret.append(row)
    for i in ret:
        for j in range(len(i), m):
            i.append(np.zeros(50))
    return np.array(ret)
