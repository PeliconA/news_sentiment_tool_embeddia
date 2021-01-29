"""

Author: Andraž Pelicon

"""

from data_transformation import prepare_data_for_prediction

import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from MLBertModelForClassification import BertClassificationTraining


def predict(data, max_len=512, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('./vocab.txt', do_lower_case=False)

    config = BertConfig.from_pretrained('../model/config.json')
    model = BertForSequenceClassification.from_pretrained('../model/pytorch_model.bin',
                                                          config=config)
    bert_trainer = BertClassificationTraining(model, device)

    dataloader = prepare_data_for_prediction(data, tokenizer, max_len, batch_size)
    predictions = bert_trainer.predict(dataloader)
    return predictions

