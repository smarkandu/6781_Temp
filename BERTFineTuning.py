import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from torch import nn, optim
import transformers as ppb
import data_collection
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from performance_metrics import print_all_metrics


class BERTFineTuning:
    def __init__(self, model_name='distilbert-base-uncased', epochs=3, lr=1e-5, batch_size=16):
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size  # Add batch_size parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.dataloader = None

    def load_data(self, df_train):
        self.df_train = df_train
        self.df_train.columns = [0, 1]
        return df_train

    def initialize_model(self):
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased'
        )

        if self.model_name == 'bert-base-uncased':
            model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights).to(self.device)  # Move model to GPU
        self.model.train()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def preprocess_data(self):
        # Tokenize the data
        tokenized = self.df_train[0].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, truncation=True))

        # Find the maximum sequence length
        max_len = max([len(x) for x in tokenized])

        # Pad sequences to the same length
        padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized], device=self.device)

        # Create attention mask
        attention_mask = (padded != 0).long().to(self.device)

        # Create labels tensor
        labels = torch.tensor(self.df_train[1].values, dtype=torch.long, device=self.device)

        # Create a dataset and DataLoader
        dataset = TensorDataset(padded, attention_mask, labels)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fine_tune_model(self):
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                input_ids, attention_mask, labels = batch

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state

                # Ensure logits are float32
                logits = last_hidden_states[:, 0, :].float()  # Convert to float32

                # Ensure labels are long (int64)
                labels = labels.to(torch.long)  # Ensure labels are long

                # Compute the loss
                loss = self.loss_fn(logits, labels)
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

    def evaluate_model(self, df_test):
        # Process the test data
        df_test.columns = [0, 1]
        tokenized = df_test[0].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, truncation=True))

        # Find the maximum sequence length
        max_len = max([len(x) for x in tokenized])

        # Pad sequences to the same length
        padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized], device=self.device)

        # Create attention mask
        attention_mask = (padded != 0).long().to(self.device)

        # Create labels tensor
        labels = torch.tensor(df_test[1].values, dtype=torch.long, device=self.device)

        # Create a dataset and DataLoader for test data
        dataset = TensorDataset(padded, attention_mask, labels)
        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Evaluate the model
        self.model.eval()
        predictions_list, labels_list = [], []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions_list.extend(predictions)
                labels_list.extend(labels.cpu().numpy())

        # Print Metrics
        print_all_metrics(labels_list, predictions_list)
        print(f"Confusion Matrix:\n{confusion_matrix(labels_list, predictions_list)}")


def get_BERT_model(df_train, batch_size_val):
    bert_finetuning = BERTFineTuning(model_name='distilbert-base-uncased', epochs=3, lr=1e-5, batch_size=batch_size_val)

    # Load data
    # _, _, _, df_train, df_test = get_data2()
    bert_finetuning.load_data(df_train)

    # Initialize model
    bert_finetuning.initialize_model()

    # Preprocess data
    bert_finetuning.preprocess_data()

    # Fine-tune the BERT model
    bert_finetuning.fine_tune_model()

    return bert_finetuning


_, _, _, df_train, df_test = data_collection.get_data2()
train_size = 1000
test_size = int(train_size*0.2)
df_train = df_train.sample(n=train_size, random_state=42)
df_test = df_test.sample(n=test_size, random_state=42)
bert_model = get_BERT_model(df_train, 8)

bert_model.evaluate_model(df_test)
