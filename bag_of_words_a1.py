import math
SMOOTHING_FACTOR = 0.2
from utils import preprocess_text

def calculate_log_likelihood(vocab, positive_vocab, negative_vocab, smoothing_factor=SMOOTHING_FACTOR):
    """ Calculates log likelihood of words belonging to a positive or negative review given a dataset and vocabulary
  Arguments
  ---------
  dataset : list of tuples
      List of positive or negative reviews with their respective label (label, text)
  vocab : dictionary
      Vocabulary of words in the dataset with their respective frequencies
  Returns
  -------
  likelihood : dictionary of dictionaries
      Dictionary of words and their positive and negative log likelihood with format {word: {'positive': log_likelihood, 'negative': log_likelihood}}
  """
    likelihood = {}
    # TODO: create a dictionary with the log likelihoods of each word
    number_instances_positive = sum(positive_vocab.values())  # number of words in positive_vocab
    number_instances_negative = sum(negative_vocab.values())  # number of words in negative_vocab
    number_types = len(vocab)  # number of words in all vocab

    for word in vocab.keys():
        likelihood[word] = {}
        # TODO: Calculate positive and negative log likelihood for EACH word.
        # IMPORTANT: remember some words might be in positives but not negatives (or the other way around, thats why we use the smoothing factor!)
        positive_log_likelihood = math.log((positive_vocab.get(word, 0) + smoothing_factor) / (
                number_instances_positive + (smoothing_factor * number_types)))
        negative_log_likelihood = math.log((negative_vocab.get(word, 0) + smoothing_factor) / (
                number_instances_negative + (smoothing_factor * number_types)))
        likelihood[word] = {'positive': positive_log_likelihood, 'negative': negative_log_likelihood}

    return likelihood


def classify_review(text, likelihood, positive_prior, negative_prior):
    """ Calculates log scores for a new text given some prior probabilities and likelihoods
    Arguments
    ---------
    text : string
        Text to classify
    likelihood_positive : dictionary
        Dictionary of words and their log likelihood for positive reviews
    likelihood_negative : dictionary
        Dictionary of words and their log likelihood for negative reviews
    positive_prior : float
        Prior probability of a review being positive
    negative_prior : float
        Prior probability of a review being negative
    Returns
    -------
    predicted sentiment : string
        Predicted sentiment of the text
    sentiment_scores : tuple or dictionary
        Tuple of positive and negative sentiment scores
    """
    tokens = preprocess_text(text).split()  # # Split the input review

    log_score_positive = math.log(positive_prior)
    log_score_negative = math.log(negative_prior)

    # Calculate the log scores for each sentiment category (take into account value for unseen tokens)
    for token in tokens:
        if token in likelihood:
            log_score_positive += likelihood[token]['positive']
            log_score_negative += likelihood[token]['negative']

    sentiment_scores = {
        'positive': log_score_positive,
        'negative': log_score_negative,
    }

    predicted_sentiment = 1 if log_score_positive > log_score_negative else 0  # Determine the predicted sentiment based on the highest sentiment score

    return predicted_sentiment, sentiment_scores
