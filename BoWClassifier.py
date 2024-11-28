import math
from collections import defaultdict

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data_collection import AI_GENERATED, HUMAN, get_data1, get_data2
from performance_metrics import print_all_metrics
from utils import preprocess_text

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
        self.ai_generated_vocab = {}
        self.number_instances_human = 0
        self.number_instances_ai_generated = 0
        self.number_types = 0

    def fit(self, vocab, human_vocab, ai_generated_vocab):
        """
        Fit the classifier with vocabulary and class-specific word counts.

        Parameters
        ----------
        vocab : dict
            Vocabulary of all words in the dataset.
        human_vocab : dict
            Vocabulary for human reviews with word frequencies.
        ai_generated_vocab : dict
            Vocabulary for ChatGPT reviews with word frequencies.
        """
        self.vocab = vocab
        self.human_vocab = human_vocab
        self.ai_generated_vocab = ai_generated_vocab
        self.number_instances_human = sum(human_vocab.values())
        self.number_instances_ai_generated = sum(ai_generated_vocab.values())
        self.number_types = len(vocab)

        self.likelihood = self._calculate_log_likelihood()

    def _calculate_log_likelihood(self):
        """
        Calculates the log likelihood of words for human and ai_generated classes.

        Returns
        -------
        dict
            Dictionary with words and their respective log likelihoods for human and ai_generated.
        """
        likelihood = defaultdict(dict)

        for word in self.vocab:
            human_count = self.human_vocab.get(word, 0)
            ai_generated_count = self.ai_generated_vocab.get(word, 0)

            # Log likelihood for human and ai_generated categories with smoothing
            likelihood[word]['human'] = math.log((human_count + self.smoothing_factor) /
                                                 (
                                                         self.number_instances_human + self.smoothing_factor * self.number_types))

            likelihood[word]['ai_generated'] = math.log((ai_generated_count + self.smoothing_factor) /
                                                   (
                                                           self.number_instances_ai_generated + self.smoothing_factor * self.number_types))

        return likelihood

    def classify(self, text, human_prior, ai_generated_prior):
        """
        Classifies a new review as either human or ai_generated based on the log likelihoods.

        Parameters
        ----------
        text : str
            The review text to classify.
        human_prior : float
            The prior probability of the review being human.
        ai_generated_prior : float
            The prior probability of the review being ChatGPT.

        Returns
        -------
        predicted_classification : int
            1 if the review is predicted as human, 0 if predicted as ChatGPT.
        classification_scores : dict
            Dictionary containing the log scores for both classes ('human' and 'ai_generated').
        """
        tokens = preprocess_text(text).split()  # Preprocess and split the review text

        log_score_human = math.log(human_prior)
        log_score_ai_generated = math.log(ai_generated_prior)

        # Calculate the log score for each word in the review (consider unseen words)
        for token in tokens:
            if token in self.likelihood:
                log_score_human += self.likelihood[token]['human']
                log_score_ai_generated += self.likelihood[token]['ai_generated']
            else:
                # If token is unseen, we apply smoothing (or skip it if no smoothing is desired)
                log_score_human += math.log(
                    self.smoothing_factor / (self.number_instances_human + self.smoothing_factor * self.number_types))
                log_score_ai_generated += math.log(
                    self.smoothing_factor / (self.number_instances_ai_generated + self.smoothing_factor * self.number_types))

        classification_scores = {
            'human': log_score_human,
            'ai_generated': log_score_ai_generated,
        }

        # Predicted classification is based on the higher log score
        predicted_class = 1 if log_score_human > log_score_ai_generated else 0

        return predicted_class, classification_scores


if __name__ == '__main__':
    full_vocab, human_vocab, ai_generated_vocab, df_train, df_test = get_data2()

    classifier = BoWClassifier(smoothing_factor=0.2)

    # Train the model with vocabularies for human and ai_generated reviews
    classifier.fit(full_vocab, human_vocab, ai_generated_vocab)

    # Classify a new review with known priors for each class
    ai_generated_prior = len(df_train[df_train['label'] == AI_GENERATED]) / len(df_train)  # calculate ai_generated prior
    human_prior = len(df_train[df_train['label'] == HUMAN]) / len(df_train)  # calculate human prior

    predicted_classifications = []
    target_classifications = df_test['label'].tolist()
    for i in range(0, df_test.shape[0]):
        predicted_classification, classification_scores = classifier.classify(df_test.iloc[i]['text'], human_prior,
                                                                              ai_generated_prior)
        predicted_classifications.append(predicted_classification)

    print_all_metrics(target_classifications, predicted_classifications)
