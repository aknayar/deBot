import pandas as pd
import numpy as np

def excel_to_arrays(filepath):
    """
    Processes the excel files of test and training data as two arrays for understandability.
    The original headings of the excel file are speaker, statement, and rating. The identity
    of the speaker is irrelevant for the purpose of this project, so that information will be discarded.
    :param
        filepath: raw str, the file path to the csv
    :return:
        statements: numpy.ndarray of size 
    """
    data = pd.read_excel(filepath)
    statements = np.array(list(data["Statement"]))
    ratings = np.array(list(data["Rating"]))
    return statements, ratings

