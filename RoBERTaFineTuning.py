import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import data_collection
from performance_metrics import print_all_metrics


class RoBERTaFineTuning:
    def __init__(self, model_name='roberta-base', epochs=3, lr=1e-5, batch_size=16):
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.dataloader = None

    def load_data(self, df_train):
        """
        Load training data.
        :param df_train: Pandas dataframe for training data
        """
        self.df_train = df_train
        self.df_train.columns = [0, 1]  # Ensure column consistency

    def initialize_model(self):
        """
        Initialize tokenizer, model, optimizer, and loss function.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)
        self.model.train()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def preprocess_data(self):
        """
        Tokenize and preprocess training data.
        """
        tokenized = self.df_train[0].apply(
            lambda x: self.tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512)
        )
        max_len = max(len(x) for x in tokenized)

        padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized], device=self.device)
        attention_mask = (padded != 0).long().to(self.device)
        labels = torch.tensor(self.df_train[1].values, dtype=torch.long, device=self.device)

        dataset = TensorDataset(padded, attention_mask, labels)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fine_tune_model(self):
        """
        Fine-tune the RoBERTa model.
        """
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                input_ids, attention_mask, labels = batch

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = self.loss_fn(logits, labels)
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

    def evaluate_model(self, df_test):
        """
        Evaluate the fine-tuned RoBERTa model on the test dataset.
        """
        df_test.columns = [0, 1]
        tokenized = df_test[0].apply(
            lambda x: self.tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512)
        )
        max_len = max(len(x) for x in tokenized)

        padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized], device=self.device)
        attention_mask = (padded != 0).long().to(self.device)
        labels = torch.tensor(df_test[1].values, dtype=torch.long, device=self.device)

        dataset = TensorDataset(padded, attention_mask, labels)
        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

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

        print_all_metrics(labels_list, predictions_list)


def get_RoBERTa_model(df_train, batch_size_val):
    roberta_finetuning = RoBERTaFineTuning(model_name='roberta-base', epochs=3, lr=1e-5, batch_size=batch_size_val)

    roberta_finetuning.load_data(df_train)
    roberta_finetuning.initialize_model()
    roberta_finetuning.preprocess_data()
    roberta_finetuning.fine_tune_model()

    return roberta_finetuning


# Example usage:
_, _, _, df_train, df_test = data_collection.get_data2()
train_size = 100
test_size = int(train_size * 0.2)
df_train = df_train.sample(n=train_size, random_state=42)
df_test = df_test.sample(n=test_size, random_state=42)
roberta_model = get_RoBERTa_model(df_train, 8)

roberta_model.evaluate_model(df_test)
