import math
from collections import defaultdict

from sklearn.model_selection import train_test_split

from utils import preprocess_text, convert_files_to_csv, vocab_dictionary

SMOOTHING_FACTOR = 0.2


class BoWClassifier:
    def __init__(self, smoothing_factor=SMOOTHING_FACTOR):
        """
        Initializes the BoW classifier with a given smoothing factor.

        Parameters
        ----------
        smoothing_factor : float, optional
            The smoothing factor used in log likelihood calculation (default is 0.2).
        """
        self.smoothing_factor = smoothing_factor
        self.likelihood = {}
        self.vocab = {}
        self.human_vocab = {}
        self.chatgpt_vocab = {}
        self.number_instances_human = 0
        self.number_instances_chatgpt = 0
        self.number_types = 0

    def fit(self, vocab, human_vocab, chatgpt_vocab):
        """
        Fit the classifier with vocabulary and class-specific word counts.

        Parameters
        ----------
        vocab : dict
            Vocabulary of all words in the dataset.
        human_vocab : dict
            Vocabulary for human reviews with word frequencies.
        chatgpt_vocab : dict
            Vocabulary for ChatGPT reviews with word frequencies.
        """
        self.vocab = vocab
        self.human_vocab = human_vocab
        self.chatgpt_vocab = chatgpt_vocab
        self.number_instances_human = sum(human_vocab.values())
        self.number_instances_chatgpt = sum(chatgpt_vocab.values())
        self.number_types = len(vocab)

        self.likelihood = self._calculate_log_likelihood()

    def _calculate_log_likelihood(self):
        """
        Calculates the log likelihood of words for human and chatgpt classes.

        Returns
        -------
        dict
            Dictionary with words and their respective log likelihoods for human and chatgpt.
        """
        likelihood = defaultdict(dict)

        for word in self.vocab:
            human_count = self.human_vocab.get(word, 0)
            chatgpt_count = self.chatgpt_vocab.get(word, 0)

            # Log likelihood for human and chatgpt categories with smoothing
            likelihood[word]['human'] = math.log((human_count + self.smoothing_factor) /
                                                 (
                                                             self.number_instances_human + self.smoothing_factor * self.number_types))

            likelihood[word]['chatgpt'] = math.log((chatgpt_count + self.smoothing_factor) /
                                                   (
                                                               self.number_instances_chatgpt + self.smoothing_factor * self.number_types))

        return likelihood

    def classify(self, text, human_prior, chatgpt_prior):
        """
        Classifies a new review as either human or chatgpt based on the log likelihoods.

        Parameters
        ----------
        text : str
            The review text to classify.
        human_prior : float
            The prior probability of the review being human.
        chatgpt_prior : float
            The prior probability of the review being ChatGPT.

        Returns
        -------
        predicted_classification : int
            1 if the review is predicted as human, 0 if predicted as ChatGPT.
        classification_scores : dict
            Dictionary containing the log scores for both classes ('human' and 'chatgpt').
        """
        tokens = preprocess_text(text).split()  # Preprocess and split the review text

        log_score_human = math.log(human_prior)
        log_score_chatgpt = math.log(chatgpt_prior)

        # Calculate the log score for each word in the review (consider unseen words)
        for token in tokens:
            if token in self.likelihood:
                log_score_human += self.likelihood[token]['human']
                log_score_chatgpt += self.likelihood[token]['chatgpt']
            else:
                # If token is unseen, we apply smoothing (or skip it if no smoothing is desired)
                log_score_human += math.log(
                    self.smoothing_factor / (self.number_instances_human + self.smoothing_factor * self.number_types))
                log_score_chatgpt += math.log(
                    self.smoothing_factor / (self.number_instances_chatgpt + self.smoothing_factor * self.number_types))

        classification_scores = {
            'human': log_score_human,
            'chatgpt': log_score_chatgpt,
        }

        # Predicted sentiment is based on the higher log score
        predicted_class = 1 if log_score_human > log_score_chatgpt else 0

        return predicted_class, classification_scores

if __name__ == '__main__':
    import pandas as pd

    chatgpt_df = convert_files_to_csv("./data/chatgpt", 0)
    human_df = convert_files_to_csv("./data/human", 1)

    df_overall = pd.concat([chatgpt_df, human_df], ignore_index=True)

    df_train, df_test = train_test_split(df_overall, test_size=0.2, random_state=42)

    full_vocab = vocab_dictionary(df_train)
    # print(len(full_vocab))
    chatgpt_vocab = vocab_dictionary(df_train[df_train[
                                                  'label'] == 0])
    human_vocab = vocab_dictionary(df_train[df_train[
                                                'label'] == 1])

    classifier = BoWClassifier(smoothing_factor=0.2)

    # Train the model with vocabularies for human and chatgpt reviews
    classifier.fit(full_vocab, human_vocab, chatgpt_vocab)

    # Classify a new review with known priors for each class
    review = "The weather today is sunny and bright"
    chatgpt_prior = len(df_train[df_train['label'] == 0]) / len(df_train)  # calculate chatgpt prior
    human_prior = len(df_train[df_train['label'] == 1]) / len(df_train)  # calculate human prior

    num_correct = 0
    for i in range(0, df_test.shape[0]):
        predicted_sentiment, sentiment_scores = classifier.classify(df_test.iloc[i]['text'], human_prior, chatgpt_prior)
        # print(predicted_sentiment)
        # print(sentiment_scores)
        if predicted_sentiment == df_test.iloc[i]['label']:
            num_correct += 1

    print(num_correct / df_test.shape[0])



    print("Predicted Sentiment:", "Human" if predicted_sentiment == 1 else "ChatGPT")
    print("Sentiment Scores:", sentiment_scores)