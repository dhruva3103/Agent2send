from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch


class BertForSequenceClassificationImproved(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense1 = nn.Linear(config.hidden_size, 512)
        self.dense2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state[:, 0, :]
        sequence_output = self.dropout(sequence_output)
        x = self.dense1(sequence_output)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(logits=logits, loss=loss)
