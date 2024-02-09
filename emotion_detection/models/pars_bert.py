import torch.nn as nn
from transformers import BertModel
from tqdm import tqdm
import time
import torch

class TextClassificationParsBert(nn.Module):
    """
    This is a PyTorch model for text classification, using the BERT (Bidirectional Encoder Representations from Transformers) architecture.
    """

    def __init__(self, config, model_path):
        """
        Initialize the TextClassificationModel.

        Args:
            config (object): Configuration object for the model.

        Attributes:
            self.bert (BertModel): BERT model pre-trained on a large text corpus.
            self.dropout (nn.Dropout): Dropout layer to prevent overfitting.
            self.classifier (nn.Linear): Classifier layer for text classification.
        """
        super(TextClassificationParsBert, self).__init__()

        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (tensor): Input token IDs for text sequences.
            attention_mask (tensor): Attention mask to focus on real tokens.
            token_type_ids (tensor): Segment token IDs for text segmentation.

        Returns:
            logits (tensor): Output logits for text classification.
        """
        # Obtain the sequence output from BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        # BERT outputs a tuple, you're interested in the pooled output which is the first element.
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits


# Defining Training Process
def train(dataloader):
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    for batch_index, batch_data in enumerate(tqdm(dataloader)):
        # Extract input features and labels from the batch_data dictionary
        input_ids = batch_data['input_ids'].to('cuda:0')
        attention_mask = batch_data['attention_mask'].to('cuda:0')
        token_type_ids = batch_data['token_type_ids'].to('cuda:0')
        targets = batch_data['targets'].to('cuda:0')

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
        input_ids = batch_data['input_ids'].to('cuda:0')
        attention_mask = batch_data['attention_mask'].to('cuda:0')
        token_type_ids = batch_data['token_type_ids'].to('cuda:0')
        targets = batch_data['targets'].to('cuda:0')

        predicted_targets = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(predicted_targets, targets)
        total_acc += (predicted_targets.argmax(1) == targets).sum().item()
        total_count += targets.size(0)
        total_loss += loss.item()
  return total_acc / total_count, total_loss / len(dataloader)