#The code that generates text based on the Star Wars Text

import time
import numpy as np
#%matplotlib notebook
import matplotlib.pyplot as plt

from collections import Counter
from collections import defaultdict


def unzip(pairs):
    """
    "unzips" of groups of items into separate lists.

    Example: pairs = [("a", 1), ("b", 2), ...] --> (("a", "b", ...), (1, 2, ...))
    """
    return tuple(zip(*pairs))


# Normalizing the data. First step of the process
def normalize(counter):
    """ Convert a `letter -> count` counter to a list
       of (letter, frequency) pairs, sorted in descending order of
       frequency.

        Parameters
        -----------
        counter : collections.Counter
            letter -> count

        Returns
        -------
        List[Tuple[str, int]]
           A list of tuples - (letter, frequency) pairs in order
           of descending-frequency

        Examples
        --------
        >>> from collections import Counter
        >>> letter_count = Counter({"a": 1, "b": 3})
        >>> letter_count
        Counter({'a': 1, 'b': 3})

        >>> normalize(letter_count)
        [('b', 0.75), ('a', 0.25)]
    """
    # <COGINST>
    total = sum(counter.values())
    return [(char, cnt / total) for char, cnt in counter.most_common()]
    # </COGINST>


from collections import defaultdict


def train_lm(text, n):
    """ Train character-based n-gram language model.

    This will learn: given a sequence of n-1 characters, what the probability
    distribution is for the n-th character in the sequence.

    For example if we train on the text:
        text = "cacao"

    Using a n-gram size of n=3, then the following dict would be returned.
    See that we *normalize* each of the counts for a given history

        {'ac': [('a', 1.0)],
         'ca': [('c', 0.5), ('o', 0.5)],
         '~c': [('a', 1.0)],
         '~~': [('c', 1.0)]}

    Tildas ("~") are used for padding the history when necessary, so that it's
    possible to estimate the probability of a seeing a character when there
    aren't (n - 1) previous characters of history available.

    So, according to this text we trained on, if you see the sequence 'ac',
    our model predicts that the next character should be 'a' 100% of the time.

    For generating the padding, recall that Python allows you to generate
    repeated sequences easily:
       `"p" * 4` returns `"pppp"`

    Parameters
    -----------
    text: str
        A string (doesn't need to be lowercased).
    n: int
        The length of n-gram to analyze.

    Returns
    -------
    Dict[str, List[Tuple[str, float]]]
      {n-1 history -> [(letter, normalized count), ...]}
    A dict that maps histories (strings of length (n-1)) to lists of (char, prob)
    pairs, where prob is the probability (i.e frequency) of char appearing after
    that specific history.

    Examples
    --------
    >>> train_lm("cacao", 3)
    {'ac': [('a', 1.0)],
     'ca': [('c', 0.5), ('o', 0.5)],
     '~c': [('a', 1.0)],
     '~~': [('c', 1.0)]}
    """
    # <COGINST>
    raw_lm = defaultdict(Counter)
    history = "~" * (n - 1)

    # count number of times characters appear following different histories
    # `raw_lm`: {history -> Counter}
    for char in text:
        raw_lm[history][char] += 1
        # slide history window to the right by one character
        history = history[1:] + char

    # create final dictionary, normalizing the counts for each history
    lm = {history: normalize(counter) for history, counter in raw_lm.items()}

    return lm
    # </COGINST>

def generate_letter(lm, history):
    """ Randomly picks letter according to probability distribution associated with
    the specified history, as stored in your language model.

    Note: returns dummy character "~" if history not found in model.

    Parameters
    ----------
    lm: Dict[str, List[Tuple[str, float]]]
        The n-gram language model.
        I.e. the dictionary: history -> [(char, freq), ...]

    history: str
        A string of length (n-1) to use as context/history for generating
        the next character.

    Returns
    -------
    str
        The predicted character. '~' if history is not in language model.
    """
    # <COGINST>
    if not history in lm:
        return "~"
    letters, probs = unzip(lm[history])
    i = np.random.choice(letters, p=probs)
    return i
    # </COGINST>


def generate_text(lm, n, user_input, nletters=100):
    """ Randomly generates `nletters` of text by drawing from
    the probability distributions stored in a n-gram language model
    `lm`.

    Parameters
    ----------
    lm: Dict[str, List[Tuple[str, float]]]
        The n-gram language model.
        I.e. the dictionary: history -> [(char, freq), ...]
    n: int
        Order of n-gram model.
    nletters: int
        Number of letters to randomly generate.

    Returns
    -------
    str
        Model-generated text.
    """
    # <COGINST>
    # This is the part in the code that I was able to change
    history = "~" * (n - 1)
    history = history + user_input
    history = history[-(n - 1):]

    text = []
    for i in range(nletters):
        c = generate_letter(lm, history)
        text.append(c)
        history = history[1:] + c
    return "".join(text)
    # </COGINST>

#the dot is look into the same directory as the folder
#Getting the information of the 1st Star Wars Movie

HP1 = "./harry_potter.pickle"
with open(HP1, "rb") as f:
    HP1 = pickle.load(f)
print(str(len(HP1)) + " character(s)")
chars = set(HP1)
print("'=' is a good pad character: ", "=" not in chars)


HP2 = "./harry_potter_2.pickle"
with open(HP2, "rb") as f:
    HP2 = pickle.load(f)
print(str(len(HP2)) + " character(s)")
chars = set(HP2)
print("'=' is a good pad character: ", "=" not in chars)


HP3 = "./harry_potter_3.pickle"
with open(HP3, "rb") as f:
    HP3 = pickle.load(f)
print(str(len(HP3)) + " character(s)")
chars = set(HP3)
print("'=' is a good pad character: ", "=" not in chars)

all_3HPs = HP1 + HP2 + HP3


#code that generates the text
t0 = time.time()
lm3 = train_lm(all_3HPs, 13)

print(generate_text(lm3, 13, "hi", 2000))



