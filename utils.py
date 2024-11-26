import os
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


def convert_files_to_csv(directory, label):
    data = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            content = preprocess_text(file.read().strip())

        data.append((content, label))

    df = pd.DataFrame(data, columns=["text", "label"])

    return df


def preprocess_text(text):
    """ Method to clean reviews from noise and standarize text across the different classes.
      The preprocessing includes converting to lowercase, removing punctuation, and removing stopwords.
      Arguments
      ---------
      text : String
         Text to clean
      Returns
      -------
      text : String
          Cleaned text
      """
    stop_words = set(stopwords.words('english'))

    text = text.lower()  # make everything lower case
    text = text.replace("\n", " ")  # remove \n characters
    text = re.sub(r'[^\w\s]', ' ', text)  # remove any punctuation or special characters
    text = re.sub(r'[\d+]', ' ', text)  # remove all numbers
    text = " ".join([word for word in text.split() if
                     word not in stop_words])  # remove all stopwords (see imports to help you with this)

    return text


def vocab_dictionary(df):
    """ Creates dictionary of frequencies based on a dataset of reviews
    Arguments
    ---------
    dataset : list of tuples
      list of tuples of the form (label, text)
    Returns
    -------
    vocab_dict : dictonary
      Dictionary of words and their frequencies with the format {word: frequency}
    """

    vocab = {}  # create empty dictionary
    # iterate through rows of df and count the frequency of words
    for row in df['text']:
        words = row.split()
        for word in words:
            word = word.strip()
            if word:
                vocab[word] = vocab.get(word, 0) + 1

    return vocab
