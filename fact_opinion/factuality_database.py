import numpy as np
import convert_to_embeddings as ce
import pickle


def stitch(embeddings, ratings):
    """
    Takes in an array of text embeddings and an array of ratings and combines them into a dictionary.
    The text embeddings will be the keys and the ratings will be the values in the dictionary.
    :param embeddings: numpy.ndarray(), shape = (M, N, 50)
    An array of text embeddings
    :param ratings: numpy.ndarray(), shape = (M,)
    An array of the ratings
    :return: dict[numpy.ndarray():int]
    """
    dictionary = {embeddings[i]: ratings[i] for i in range(len(embeddings))}
    return dictionary


def create_database(statements, ratings):
    """
    Takes in an array of embeds and the array of ratings. Creates a database that is a dictionary
    that has the embeds as keys and the ratings as values.
    :param statements: numpy.ndarray(), shape = (M,)
    An array of statements.
    :param ratings: numpy.ndarray(), shape = (M,)
    An array of the ratings
    :return:
    """
    dictionary = stitch(ce.to_embeddings(statements), ratings)
    pickle_out = open("factual.pickle", "wb")
    pickle.dump(dictionary, pickle_out)
    pickle_out.close()
