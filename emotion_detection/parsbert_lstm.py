import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import pandas as pd
import time
import torch
import numpy as np
import re
from transformers import BertConfig, BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_len = 128
train_batch_size = 128
valid_batch_size = 128
test_batch_size = 128
epoch = 3
EEVERY_EPOCH = 1000
lr = 2e-5
CLIP = 0.0

data = pd.read_csv('data/train_process.csv')
data.dropna(inplace=True)

class CallCenterDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, contexts, targets=None, label_list=None, max_len=128):
        self.contexts = contexts
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Create a label-to-index mapping if label_list is provided
        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, item):
        context = str(self.contexts[item])
        if self.has_target:
            target = self.label_map.get(str(self.targets[item]), str(self.targets[item]))
        encoding = self.tokenizer.encode_plus(
            context,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')

        inputs = {
            'context': context,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }
        if self.has_target:
            inputs['targets'] = torch.tensor(target, dtype=torch.long)
        return inputs

def create_data_loader(x, y, tokenizer, max_len, batch_size, label_list):
    dataset = CallCenterDataset(
        contexts=x,
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len,
        label_list=label_list)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

def split_data(df):
    labels_list = list(sorted(df['label'].unique()))
    train, test = train_test_split(df, test_size=0.1, random_state=1, stratify=df['label'])
    train, valid = train_test_split(train, test_size=0.1, random_state=1, stratify=train['label'])
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, valid, test
train, valid, test = split_data(data)

x_train, y_train = train['text'].values.tolist(), train['label'].values.tolist()
x_valid, y_valid = valid['text'].values.tolist(), valid['label'].values.tolist()
x_test, y_test = test['text'].values.tolist(), test['label'].values.tolist()

parsbert_id = 'HooshvareLab/bert-fa-base-uncased'
label_list = list(sorted(data['label'].unique()))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}

tokenizer = BertTokenizer.from_pretrained(parsbert_id)
bertcofig = BertConfig.from_pretrained(
    parsbert_id, **{
        'label2id': label2id,
        'id2label': id2label,
    })
train_data_loader = create_data_loader(train['text'].to_numpy(), train['label'].to_numpy(), tokenizer, max_len, train_batch_size, label_list)
valid_data_loader = create_data_loader(valid['text'].to_numpy(), valid['label'].to_numpy(), tokenizer, max_len, valid_batch_size, label_list)
test_data_loader = create_data_loader(test['text'].to_numpy(), test['label'].to_numpy(), tokenizer, max_len, test_batch_size, label_list)

class TextClassificationParsBert(nn.Module):
    def __init__(self, model_path=parsbert_id, num_labels=2, hidden_size=768, num_layers=1):
        super(TextClassificationParsBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)  # Adjust dropout rate as needed
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_labels)  # Adjust output dimension
        self.softmax = nn.Softmax(dim=1)
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Use the last layer's hidden states as features
        hidden_states = outputs.last_hidden_state
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(hidden_states)

        # Apply attention mechanism
        attention_weights = self.softmax(torch.matmul(lstm_out, lstm_out.transpose(-1, -2)))
        context_vector = torch.matmul(attention_weights, lstm_out)

        # Aggregate the context vector using mean or max pooling
        pooled_output, _ = torch.max(context_vector, dim=1)

        # Pass through fully connected layer
        logits = self.fc(pooled_output)
        
        return logits


model = TextClassificationParsBert()
model = model.to(device)

def train(dataloader):
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    for batch_index, batch_data in enumerate(tqdm(dataloader)):
        # Extract input features and labels from the batch_data dictionary
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        token_type_ids = batch_data['token_type_ids'].to(device)
        targets = batch_data['targets'].to(device)

        optimizer.zero_grad()
        predicted_targets = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(predicted_targets, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_targets.argmax(1) == targets).sum().item()
        total_count += targets.size(0)
        total_loss += loss.item()

        if batch_index % log_interval == 0 and batch_index > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f} | loss {:8.3f}".format(
                    epoch, batch_index, len(dataloader), total_acc / total_count, total_loss
                )
            )
            total_acc, total_count ,total_loss = 0, 0, 0
            start_time = time.time()


def evaluate(dataloader):
  model.eval()
  total_acc, total_count ,total_loss = 0, 0, 0

  with torch.no_grad():
    for batch_index, batch_data in enumerate(tqdm(dataloader)):
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        token_type_ids = batch_data['token_type_ids'].to(device)
        targets = batch_data['targets'].to(device)

        predicted_targets = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(predicted_targets, targets)
        total_acc += (predicted_targets.argmax(1) == targets).sum().item()
        total_count += targets.size(0)
        total_loss += loss.item()
  return total_acc / total_count, total_loss / len(dataloader)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

for epoch in tqdm(range(1, 20 + 1), desc="Epochs... "):
    epoch_start_time = time.time()
    train(train_data_loader)
    accu_train, loss_train =  evaluate(train_data_loader)
    accu_val, loss_val = evaluate(valid_data_loader)



    if total_accu is not None and total_accu > accu_val:
        scheduler.step()

    else:
        total_accu = accu_val

    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        " | train accuracy {:8.3f} | loss train {:8.3f} | valid accuracy {:8.3f} | loss validation {:8.3f}".format(
            epoch, time.time() - epoch_start_time, accu_train, loss_train, accu_val, loss_val
        )
    )
    print("-" * 59)