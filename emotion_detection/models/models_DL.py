import torch
import torch.nn as nn
import time


class TextClassificationLinear(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationLinear, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class TextClassificationRNN(nn.Module):
    """
        model = TextClassificationRNN(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx).to(device)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(TextClassificationRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        output, hidden = self.rnn(embedded)
        # output = [batch size, sent len, hid dim]
        # hidden = [1, batch size, hid dim]
        assert torch.equal(output[:, -1, :], hidden.squeeze(0))
        # Get the last hidden state to pass it to the fully connected layer
        hidden = hidden.squeeze(0)
        # hidden = [batch size, hid dim]
        return self.fc(hidden)


class TextClassificationLSTM(nn.Module):
    """
        model = TextClassificationLSTM(vocab_size, embedd_dim, hidden_dim, output_dim, pad_idx).to(device)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(TextClassificationLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [batch size, sentence length]
        embedded = self.embedding(text)
        # embedded = [batch size, sentence length, embedding dim]
        output, (hidden, _) = self.lstm(embedded)
        # output = [batch size, sentence length, hidden dim]
        # hidden = [1, batch size, hidden dim]
        hidden = hidden.squeeze(0)
        # hidden = [batch size, hidden dim]
        out = self.fc(hidden)
        return out
    

class TextClassificationGRU(nn.Module):
    """
        model = TextClassificationGRU(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(TextClassificationGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [batch size, sentence length]
        embedded = self.embedding(text)
        # embedded = [batch size, sentence length, embedding dim]
        output, hidden = self.gru(embedded)
        # output = [batch size, sentence length, hidden dim]
        # hidden = [1, batch size, hidden dim]
        hidden = hidden.squeeze(0)
        # hidden = [batch size, hidden dim]
        out = self.fc(hidden)
        return out

class TextClassificationLSTMBidirectional(nn.Module):
    """
        model = TextClassificationLSTMBidirectional(INPUT_DIM,
                          EMBEDDING_DIM,
                          HIDDEN_DIM,
                          OUTPUT_DIM,
                          N_LAYERS,
                          BIDIRECTIONAL,
                          DROPOUT,
                          PAD_IDX).to(device)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        #embedded = [batch size, sent len, emb dim]
        output, (hidden, cell) = self.lstm(embedded)
        #output = [batch size, sent len, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        #concat the final forward and backward hidden state
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        #hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)

def train(dataloader):
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (text, label) in enumerate(dataloader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text)
        # Ensure label is of correct shape and type
        label = label.squeeze()  # Remove any extra dimensions
        label = label.long()

        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        total_loss += loss.item()

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f} | loss {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count, total_loss
                )
            )
            total_acc, total_count ,total_loss = 0, 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count ,total_loss = 0, 0, 0
    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):

            text, label = text.to(device), label.to(device)
            # Ensure label is of correct shape and type
            label = label.squeeze()  # Remove any extra dimensions
            label = label.long()

            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.item()
    return total_acc / total_count, total_loss / len(dataloader)