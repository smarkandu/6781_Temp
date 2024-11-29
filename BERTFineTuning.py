import torch
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification
from torch import nn, optim
import transformers as ppb
import data_collection
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from performance_metrics import print_all_metrics


class BERTFineTuning:
    def __init__(self, model_name='distilbert-base-uncased', epochs=3, lr=1e-5, batch_size=16):
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
        We insert our training data using this method
        :param df_train: Pandas dataframe for training data
        :return: Modified pandas dataframe for training data
        """
        self.df_train = df_train
        self.df_train.columns = [0, 1]
        return df_train

    def initialize_model(self):
        """
        Initialize Data Members of object
        :return: None
        """
        if self.model_name == 'bert-base-uncased':
            model_class, tokenizer_class, pretrained_weights = (
                BertForSequenceClassification, ppb.BertTokenizer, 'bert-base-uncased')
        else:
            model_class, tokenizer_class, pretrained_weights = (
                DistilBertForSequenceClassification, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Set up tokenizer and model using pre-trained weights
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights, num_labels=2).to(self.device)
        self.model.train()  # Put model in training mode

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def preprocess_data(self):
        """
        We tokenize and pad our data here.  We also create the attention mask, dataset and dataloader
        :return: None
        """
        tokenized = self.df_train[0].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, truncation=True))
        max_len = max([len(x) for x in tokenized])

        padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized], device=self.device)
        attention_mask = (padded != 0).long().to(self.device)
        labels = torch.tensor(self.df_train[1].values, dtype=torch.long, device=self.device)

        dataset = TensorDataset(padded, attention_mask, labels)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fine_tune_model(self):
        """
        Fine-Tune our model by training on a few epochs of our training data for our downstream task (classification)
        :return: None
        """
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                input_ids, attention_mask, labels = batch

                self.optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute loss
                labels = labels.to(torch.long)
                loss = self.loss_fn(logits, labels)
                epoch_loss += loss.item()

                # Perform the backwards pass
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

    def evaluate_model(self, df_test):
        df_test.columns = [0, 1]  # Set Column Names
        tokenized = df_test[0].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, truncation=True))
        max_len = max([len(x) for x in tokenized]) # Obtain the max size of a 'text' encoding

        # Add padding to the tokenized 'text' sequences
        # Create the attention mask in order for the model to know which tokens to pay attention to
        # (i.e. real tokens vs padding)
        padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized], device=self.device)
        attention_mask = (padded != 0).long().to(self.device)
        labels = torch.tensor(df_test[1].values, dtype=torch.long, device=self.device)

        # Create Dataset and Data Loader for test data
        dataset = TensorDataset(padded, attention_mask, labels)
        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval() # Put model in test mode
        predictions_list, labels_list, probabilities_list = [], [], []
        with torch.no_grad():
            # Run through each batch of test data
            for batch in test_dataloader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # Compute probabilities using softmax
                probabilities = F.softmax(logits, dim=1).cpu().numpy()
                # Get predicted class probabilities for the positive class (i.e. 1)
                positive_class_probs = probabilities[:, 1]
                # Store probabilities for ROC computation
                probabilities_list.extend(positive_class_probs)
                # Get predicted labels
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions_list.extend(predictions)
                labels_list.extend(labels.cpu().numpy())

        # Print all performance Metrics observed
        basic_metrics, adv_metrics = print_all_metrics(labels_list, predictions_list)

        return basic_metrics, adv_metrics, predictions_list, labels_list


def get_BERT_model(df_train, batch_size_val):
    bert_finetuning = BERTFineTuning(model_name='distilbert-base-uncased', epochs=3, lr=1e-5, batch_size=batch_size_val)

    # Load data
    bert_finetuning.load_data(df_train)

    # Initialize model
    bert_finetuning.initialize_model()

    # Preprocess data
    bert_finetuning.preprocess_data()

    # Fine-tune the BERT model
    bert_finetuning.fine_tune_model()

    return bert_finetuning


# _, _, _, df_train, df_test = data_collection.get_data_primary(data_collection.get_overall_data_primary())
# train_size = 100
# test_size = int(train_size * 0.2)
# df_train = df_train.sample(n=train_size, random_state=42)
# df_test = df_test.sample(n=test_size, random_state=42)
# bert_model = get_BERT_model(df_train, 8)
#
# bert_model.evaluate_model(df_test)
