import os
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

torch.manual_seed(0)

nltk.download('stopwords')


def convert_text_files_to_df(directory, label):
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


def word_to_index(vocabulary):
    """ Method to create vocabulary to index mapping.
    Arguments
    ---------
    vocabulary : Dictionary
       Dictonary of format {word:frequency}
    Returns
    -------
    word_to_index : Dictionary
        Dictionary mapping words to index with format {word:index}
    """
    # Create key,value pair for out of vocabulary worlds
    # TODO
    return_value = {'<OOV>': 0}
    for index, word in enumerate(vocabulary, start=1):
        return_value[word] = index

    return return_value


def save_at_checkpoint(model, optimizer, loss, filename, epoch_number):
    checkpoint = {
        'epoch': epoch_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    print(f"Saving training from epoch {epoch_number} using file {filename}")
    torch.save(checkpoint, filename)


def load_at_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Continuing training from epoch {epoch} using file {filename}")
    return epoch

def generate_dataset(data, window_size,word_to_index):
  """ Method to generate training dataset for CBOW.
  Arguments
  ---------
  data : String
     Training dataset
  window_size : int
     Size of the context window
  word_to_index : Dictionary
     Dictionary mapping words to index with format {word:index}
  Returns
  -------
  surroundings : N x W Tensor
      Tensor with index of surrounding words, with N being the number of samples and W being the window size
  targets : Tensor
      Tensor with index of target word
  """
  surroundings= []
  targets = []
  data= data.split(" ")
  #TODO complete function
  for i in range(window_size,len(data)-window_size):
    surrounding =  [word_to_index[surrounding_word] for surrounding_word in data[i - window_size: i] + data[i + 1: i + window_size + 1]]  #get surrounding words based on window size
    target = word_to_index[data[i]] #get target word (middle word)
    surroundings.append(surrounding) #append to surrounding
    targets.append(target) #append to targets

  surroundings = torch.tensor(surroundings)
  targets = torch.tensor(targets)

  return surroundings, targets

def plot_ROC(labels_list, probabilities_list):
    # Compute FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(labels_list, probabilities_list)

    # Compute AUC
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
