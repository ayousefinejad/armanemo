import torch.nn as nn
from transformers import BertModel


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
