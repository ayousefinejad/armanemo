from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from collections import defaultdict

# create custom dataset 
class CustomTextDataset(Dataset):
    def __init__(self, txt, labels,tokens_id):
        self.labels = labels
        self.text = txt
        self.tokens_id = tokens_id

    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        tokens_id = self.tokens_id[idx]
        sample = {"Text": text, "Class": label, "Tokens_id": tokens_id}
        return sample




def create_data_loader(train, valid, test, label2id, batch_size, device):
    
    def collate_batch(batch):
        """
        convert text and labels to tensor numbers when using DataLoader
        offset store the length of each sentence to determine the end of sentences
        input:
            batch data
        output:
            text, classes, offset
        """
        text_list, classes, offsets = [], [], [0]

        for sample in batch:
            _class = sample['Class']
            _text_ids = sample['Tokens_id']
            proceed_text = torch.tensor(_text_ids, dtype=torch.int64)
            text_list.append(proceed_text)
            classes.append(label2id[_class])
            offsets.append(proceed_text.size(0))

        text = torch.cat(text_list)
        classes = torch.tensor(classes, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        return classes.to(device), text.to(device), offsets.to(device)
    
    text_labels_df_train = pd.DataFrame({'Text': train['text'].to_numpy(), 'Labels': train['label'].to_numpy(), 
                               'Tokens_id': train['tokens_ids'].to_numpy()})
    TD_train = CustomTextDataset(text_labels_df_train['Text'], text_labels_df_train['Labels'], text_labels_df_train['Tokens_id'])

    text_labels_df_valid = pd.DataFrame({'Text': valid['text'].to_numpy(), 'Labels': valid['label'].to_numpy(), 
                                'Tokens_id': valid['tokens_ids'].to_numpy()})
    TD_valid = CustomTextDataset(text_labels_df_valid['Text'], text_labels_df_valid['Labels'], text_labels_df_valid['Tokens_id'])

    text_labels_df_test = pd.DataFrame({'Text': test['text'].to_numpy(), 'Labels': test['label'].to_numpy(), 
                                'Tokens_id': test['tokens_ids'].to_numpy()})
    TD_test = CustomTextDataset(text_labels_df_test['Text'], text_labels_df_test['Labels'], text_labels_df_test['Tokens_id'])


    train_loader = DataLoader(TD_train, batch_size=batch_size, collate_fn=collate_batch)
    valid_loader = DataLoader(TD_valid, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(TD_test, batch_size=batch_size, collate_fn=collate_batch)

    return train_loader, valid_loader, test_loader


def tokens_ids(train, valid, test, max_vocab):
  token_freq = defaultdict(int)

  for example in train.tokens:
      for token in example:
          token_freq[token] += 1

  print('maximum vocab numbers: ', max(list(token_freq.values())))

  token_freq = {key: value for key, value in token_freq.items() if value >= 3}
  sorted_tokens = ['<pad>', '<unk>'] + sorted(token_freq.keys(), key=lambda token: token_freq[token], reverse=True)[:max_vocab - 2]
  token2id = {token: idx for idx, token in enumerate(sorted_tokens)}

  def tokenize_function(example):
    """"
    Tokenize the input tokens and map them to corresponding token IDs.

    Args:
      example (dict): a dictionary containing 'tokens' to be tokenized.

    Returns:
      dict: dictionary containing 'token_ids' with token IDs.

    """
    return [token2id.get(token, token2id['<unk>']) for token in example]

  train['tokens_ids'] = train.tokens.apply(tokenize_function)
  valid['tokens_ids'] = train.tokens.apply(tokenize_function)
  test['tokens_ids'] = train.tokens.apply(tokenize_function)

  return train, valid, test, token2id

