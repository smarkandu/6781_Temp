import torch
import torch.nn as nn
import os
from tqdm import tqdm

from data_collection import get_data1
from utils import load_at_checkpoint, save_at_checkpoint, word_to_index


class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim=300):
        """ Class to define the CBOW model
    Attributes
    ---------
    device : device
      Device where the model will be trained (gpu preferably)
    vocab_size : int
      Size of the vocabulary
    embed_dim : int
      Size of the embedding layer
    hidden_dim : int
      Size of the hidden layer
    """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # use this layer to get a vector from the the word index
        self.linear = nn.Linear(embed_dim, vocab_size)  # first fully connected layer (bottleneck)

    def forward(self, x):
        emb = self.embedding(x)  # pass input through embedding layer
        average = torch.mean(emb, dim=1)  # average and resize (size must be batch_size x embed_dim)
        out = self.linear(average)  # pass through linear layer

        return out


def train_CBOW(word_to_index_dict, train_dataloader):
  device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")  # CAUTION: RUN THIS CODE WITH GPU, CPU WILL TAKE TOO LONG
  model = CBOW(len(word_to_index_dict)).to(device)
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  epochs = 10
  starting_epoch = 0
  pt_filename = "CBOW_model.pt"
  check_for_file = True

  if os.path.exists(pt_filename) and check_for_file:
    starting_epoch = load_at_checkpoint(pt_filename, model, optimizer) + 1

  # BE PATIENT: This code can take up to 2 hours and 10 min for a batch size of 64 and 10 epochs
  for epoch in range(starting_epoch, epochs):
    total_loss = 0
    for surr, tar in tqdm(train_dataloader):
      # TODO: create code for training our model
      surr, tar = surr.to(device), tar.to(device)
      optimizer.zero_grad()

      log_probs = model(surr)

      loss = loss_function(log_probs, tar)
      total_loss += loss.item()

      loss.backward()
      optimizer.step()

    save_at_checkpoint(model, optimizer, loss, pt_filename, epoch)
    print(f"Epoch {epoch + 1} loss: {total_loss / len(train_dataloader)}")

    return model


full_vocab, _, _ = get_data1()
word_to_index_dict = word_to_index(full_vocab)
model = train_CBOW(word_to_index_dict)