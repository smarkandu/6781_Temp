import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import re

from data_collection import get_data2


class TFIDFLinearRegression:
    def __init__(self):
        self.vocabulary = None
        self.idf = None
        self.model = LinearRegression()

    @staticmethod
    def preprocess(text):
        """
        Preprocess the input text by cleaning and normalizing it.

        Parameters:
        text (str): The input text document.

        Returns:
        str: The cleaned and preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit(self, documents, y):
        """
        Train the model using the given documents and target values.

        Parameters:
        documents (list of str): The list of text documents.
        y (numpy array): The target variable for each document.
        """
        # Step 0: Preprocess text
        documents = [self.preprocess(doc) for doc in documents]

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
        print(f"R² on training data: {self.model.score(X_train, y_train):.4f}")
        print(f"R² on test data: {self.model.score(X_test, y_test):.4f}")

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
            idf[word] = math.log(N / (df + 1)) + 1  # Add 1 to prevent division by zero
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
        # Preprocess the input documents
        documents = [self.preprocess(doc) for doc in documents]
        tf_matrix = [self.compute_tf(doc) for doc in documents]
        tfidf_matrix = self.compute_tfidf(tf_matrix)
        return self.model.predict(tfidf_matrix)


# Example usage:
# Sample data
documents = [
    "This is a sample document.",
    "This document is another example of a document.",
    "Sample text for testing.",
    "Testing documents is essential."
]
y = np.array([1, 0, 1, 0])  # Dummy target for demonstration

full_vocab, human_vocab, ai_generated_vocab, df_train, df_test = get_data2()

# Instantiate the model
model = TFIDFLinearRegression()

# Fit the model to the data
model.fit(df_test['text'].tolist(), df_test['label'].tolist())

# Predict on new data
predictions = model.predict(["this is a test document", "new sample document for testing"])

print(f"Predictions: {predictions}")
