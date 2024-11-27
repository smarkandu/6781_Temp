import math
SMOOTHING_FACTOR = 0.2
from utils import preprocess_text

def calculate_log_likelihood(vocab, human_vocab, chatgpt_vocab, smoothing_factor=SMOOTHING_FACTOR):
    """ Calculates log likelihood of words belonging to a human or chatgpt review given a dataset and vocabulary
  Arguments
  ---------
  dataset : list of tuples
      List of human or chatgpt reviews with their respective label (label, text)
  vocab : dictionary
      Vocabulary of words in the dataset with their respective frequencies
  Returns
  -------
  likelihood : dictionary of dictionaries
      Dictionary of words and their human and chatgpt log likelihood with format {word: {'human': log_likelihood, 'chatgpt': log_likelihood}}
  """
    likelihood = {}
    number_instances_human = sum(human_vocab.values())  # number of words in human_vocab
    number_instances_chatgpt = sum(chatgpt_vocab.values())  # number of words in chatgpt_vocab
    number_types = len(vocab)  # number of words in all vocab

    for word in vocab.keys():
        likelihood[word] = {}
        # IMPORTANT: remember some words might be in humans but not chatgpts (or the other way around, thats why we use the smoothing factor!)
        human_log_likelihood = math.log((human_vocab.get(word, 0) + smoothing_factor) / (
                number_instances_human + (smoothing_factor * number_types)))
        
        chatgpt_log_likelihood = math.log((chatgpt_vocab.get(word, 0) + smoothing_factor) / (
                number_instances_chatgpt + (smoothing_factor * number_types)))
        
        likelihood[word] = {'human': human_log_likelihood, 'chatgpt': chatgpt_log_likelihood}

    return likelihood


def classify_review(text, likelihood, human_prior, chatgpt_prior):
    """ Calculates log scores for a new text given some prior probabilities and likelihoods
    Arguments
    ---------
    text : string
        Text to classify
    likelihood_human : dictionary
        Dictionary of words and their log likelihood for human reviews
    likelihood_chatgpt : dictionary
        Dictionary of words and their log likelihood for chatgpt reviews
    human_prior : float
        Prior probability of a review being human
    chatgpt_prior : float
        Prior probability of a review being chatgpt
    Returns
    -------
    predicted classification : string
        Predicted classification of the text
    classification_scores : tuple or dictionary
        Tuple of human and chatgpt classification scores
    """
    tokens = preprocess_text(text).split()  # # Split the input review

    log_score_human = math.log(human_prior)
    log_score_chatgpt = math.log(chatgpt_prior)

    # Calculate the log scores for each classification category (take into account value for unseen tokens)
    for token in tokens:
        if token in likelihood:
            log_score_human += likelihood[token]['human']
            log_score_chatgpt += likelihood[token]['chatgpt']

    classification_scores = {
        'human': log_score_human,
        'chatgpt': log_score_chatgpt,
    }

    predicted_classification = 1 if log_score_human > log_score_chatgpt else 0  # Determine the predicted classification based on the highest classification score

    return predicted_classification, classification_scores
