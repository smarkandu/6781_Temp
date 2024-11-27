import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_collection import get_data2


class TFIDFLinearRegression:
    def __init__(self):
        self.vocabulary = None
        self.idf = None
        self.model = LinearRegression()

    def fit(self, documents, y):
        """
        Train the model using the given documents and target values.

        Parameters:
        documents (list of str): The list of text documents.
        y (numpy array): The target variable for each document.
        """
        # Step 1: Build vocabulary
        self.vocabulary = list(set(' '.join(documents).split()))

        # Step 2: Compute TF (Term Frequency)
        tf_matrix = [self.compute_tf(doc) for doc in documents]

        # Step 3: Compute IDF (Inverse Document Frequency)
        self.idf = self.compute_idf(documents)

        # Step 4: Compute TF-IDF matrix
        tfidf_matrix = self.compute_tfidf(tf_matrix)

        # Step 5: Train Linear Regression model
        X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        print(f"Model trained with {len(documents)} documents.")

    def compute_tf(self, doc):
        """
        Compute Term Frequency (TF) for a document.

        Parameters:
        doc (str): A single document as a string.

        Returns:
        dict: A dictionary with words and their respective TF values.
        """
        tf = {}
        words = doc.split()
        total_words = len(words)
        for word in self.vocabulary:
            tf[word] = words.count(word) / total_words
        return tf

    def compute_idf(self, documents):
        """
        Compute Inverse Document Frequency (IDF) for each word in the vocabulary.

        Parameters:
        documents (list of str): The list of text documents.

        Returns:
        dict: A dictionary with words and their respective IDF values.
        """
        idf = {}
        N = len(documents)
        for word in self.vocabulary:
            df = sum(1 for doc in documents if word in doc)
            idf[word] = math.log(N / (df + 1)) + 1  # add 1 to prevent division by zero
        return idf

    def compute_tfidf(self, tf_matrix):
        """
        Compute TF-IDF matrix from TF and IDF values.

        Parameters:
        tf_matrix (list of dict): A list of dictionaries, each representing a document's term frequency.

        Returns:
        numpy array: The TF-IDF matrix.
        """
        tfidf_matrix = []
        for tf in tf_matrix:
            tfidf_vector = [tf[word] * self.idf[word] for word in self.vocabulary]
            tfidf_matrix.append(tfidf_vector)
        return np.array(tfidf_matrix)

    def predict(self, documents):
        """
        Predict the target values for a list of documents.

        Parameters:
        documents (list of str): The list of text documents.

        Returns:
        numpy array: Predicted values.
        """
        tf_matrix = [self.compute_tf(doc) for doc in documents]
        tfidf_matrix = self.compute_tfidf(tf_matrix)
        return self.model.predict(tfidf_matrix)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance using test data.

        Parameters:
        X_test (numpy array): The test features (TF-IDF).
        y_test (numpy array): The actual target values.

        Returns:
        float: The R^2 score of the model.
        """
        return self.model.score(X_test, y_test)


# Example usage:
full_vocab, human_vocab, chatgpt_vocab, df_train, df_test = get_data2()


documents = [
    "this is a sample document",
    "this document is a sample",
    "sample document for testing",
    "testing document example"
]
y = np.array([1, 0, 1, 0])  # Dummy target for demonstration

# Instantiate the model
model = TFIDFLinearRegression()

# Fit the model to the data
model.fit(df_test['text'].tolist(), df_test['label'].tolist())

# Predict on new data
predictions = model.predict(["this is a test document", "new sample document for testing"])

print(f"Predictions: {predictions}")
