from nltk import tokenize
from textstat.textstat import textstatistics, legacy_round
import textstat

# Splits the text into sentences
def break_sentences(text):
    return tokenize.sent_tokenize(text)

# Returns Number of Words in the text
def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words

# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)

# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length

# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
    return textstatistics().syllable_count(word)

# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)

def flesch_reading_ease(text):
    """
        Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        Here,
          ASL = average sentence length (number of words
                divided by number of sentences)
          ASW = average word length in syllables (number of syllables
                divided by number of words)
        The maximum possible score is 121.22. Negative scores are valid.
    """
    return textstat.flesch_reading_ease(text)

def flesch_kincaid_grade(text):
    """
    Implements the Flesch-Kincaid reading level formula:
    Flesh-Kincaid Grade Level = 0.39*ASL + 11.8*ASW - 15.59
    Here,
        ASL = average sentence length (number of words divided by number of sentences)
        ASW = average word length in syllables (number of syllables divided by number of words)
    :param text: The text
    :return: a grade level from 0-18, where 0 is the easiest, and 18 is the hardest.
    The goal is to aim for a score of 8 to ensure that 80% of American can read/understand it.
    """
    return textstat.flesch_kincaid_grade(text)
