import torch
import numpy as np
from sklearn.model_selection import train_test_split

class CallCenterDataset(torch.utils.data.Dataset):
    """
    This is a PyTorch dataset class designed for processing text data, commonly used in a call center context.
    """

    def __init__(self, tokenizer, contexts, targets=None, label_list=None, max_len=128):
        """
        Initialize the dataset.

        Args:
            tokenizer (object): A tokenizer for encoding text data.
            contexts (list): A list of text data representing call center conversations or context.
            targets (list or None): A list of labels associated with each context, if available.
            label_list (list or None): A list of unique labels if targets are provided.
            max_len (int): The maximum length of the tokenized sequences.

        Attributes:
            self.contexts (list): List of input text data.
            self.targets (list or None): List of labels corresponding to each input context.
            self.has_target (bool): True if labels are provided, False otherwise.
            self.tokenizer (object): Tokenizer for text data.
            self.max_len (int): Maximum length for tokenized sequences.
            self.label_map (dict): A mapping from labels to integer indices, if label_list is provided.
        """
        self.contexts = contexts
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Create a label-to-index mapping if label_list is provided
        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.contexts)

    def __getitem__(self, item):
        """
        Get a specific item (sample) from the dataset.

        Args:
            item (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing various data elements for the specified item.
        """
        context = str(self.contexts[item])

        if self.has_target:
            target = self.label_map.get(str(self.targets[item]), str(self.targets[item]))

        # Tokenize the input text and format it for model input
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
    """
    Create a PyTorch DataLoader for a given dataset.

    Args:
        x (list): List of input text data.
        y (list or None): List of labels corresponding to each input context.
        tokenizer (object): Tokenizer for text data.
        max_len (int): Maximum length for tokenized sequences.
        batch_size (int): Batch size for DataLoader.
        label_list (list): List of unique labels.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = CallCenterDataset(
        contexts=x,
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len,
        label_list=label_list)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def split_data(df, target_name):
    """
    Split the dataset into training, validation, and test sets, and prepare input features and labels.

    Args:
        df (DataFrame): Pandas DataFrame containing columns label and text.

    Returns:
        tuple: A tuple containing x_train, y_train, x_valid, y_valid, x_test, and y_test.
    """
    labels_list = list(sorted(df[target_name].unique()))
    df['label_id'] = df[target_name].apply(lambda t: labels_list.index(t))

    train, test = train_test_split(df, test_size=0.1, random_state=1, stratify=df[target_name])
    train, valid = train_test_split(train, test_size=0.1, random_state=1, stratify=train[target_name])

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, valid, test