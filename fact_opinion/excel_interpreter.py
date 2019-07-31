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
        statements: numpy.ndarray of size (M,)
                    An array of all the statements
        ratings:    numpy.ndarray of size (M,)
                    An array of all the ratings
    """
    data = pd.read_excel(filepath, sheet_name="Sheet1")
    statements = np.array(list(data["Statement"]))
    ratings = np.array(list(data["Rating"]))
    return statements, ratings
