import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from torch import nn, optim
import transformers as ppb
from data_collection import get_data1


class BERTFineTuning:
    def __init__(self, model_name='distilbert-base-uncased', epochs=3, lr=1e-5):
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def load_data(self):
        full_vocab, human_vocab, chatgpt_vocab, df_train, df_test = get_data1()
        self.df_train = df_train
        self.df_test = df_test
        return df_train, df_test

    def initialize_model(self):
        model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        if self.model_name == 'bert-base-uncased':
            model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        self.model.train()  # Set the model in training mode

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def preprocess_data(self):
        # Tokenize the data
        tokenized = self.df_train[0].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, truncation=True))

        # Find the maximum sequence length
        max_len = max([len(x) for x in tokenized])

        # Pad sequences to the same length
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])

        # Create attention mask
        attention_mask = np.where(padded != 0, 1, 0)

        # Convert to tensors
        self.input_ids = torch.tensor(padded)
        self.attention_mask = torch.tensor(attention_mask)

    def fine_tune_model(self):
        labels = self.df_train[1].values

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(self.input_ids, attention_mask=self.attention_mask)
            last_hidden_states = outputs[0]  # Get hidden states
            logits = last_hidden_states[:, 0, :]  # [CLS] token representation

            # Compute the loss
            loss = self.loss_fn(logits, torch.tensor(labels).long())

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

    def evaluate_model(self):
        # Extract features after fine-tuning
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(self.input_ids, attention_mask=self.attention_mask)
            last_hidden_states = outputs[0]
            features = last_hidden_states[:, 0, :].numpy()

        # Prepare data for classification
        labels = self.df_train[1]
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

        # Train a Logistic Regression classifier on the features
        lr_clf = LogisticRegression()
        lr_clf.fit(train_features, train_labels)

        # Evaluate the classifier
        score = lr_clf.score(test_features, test_labels)
        print(f'Logistic Regression Score: {score}')


# Usage example:
if __name__ == "__main__":
    bert_finetuning = BERTFineTuning(model_name='distilbert-base-uncased', epochs=3, lr=1e-5)

    # Load data
    df_train, df_test = bert_finetuning.load_data()

    # Initialize model
    bert_finetuning.initialize_model()

    # Preprocess data
    bert_finetuning.preprocess_data()

    # Fine-tune the BERT model
    bert_finetuning.fine_tune_model()

    # Evaluate the model
    bert_finetuning.evaluate_model()
