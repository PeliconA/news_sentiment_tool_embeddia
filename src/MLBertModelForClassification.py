"""

Author: Andraž Pelicon

"""

import torch
#from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
#from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import os
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
#from sklearn.model_selection import train_test_split
#from preprocess import transform_data


class BertClassificationTraining():
    """Finetunes multilingual BERT model on a classification task"""

    def __init__(self, model, device, batch_size=32, lr=2e-5, train_epochs=3, weight_decay=0.01,
                 warmup_proportion=0.1, adam_epsilon=1e-8):
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = lr
        self.train_epochs = train_epochs
        self.weight_decay = weight_decay
        self.warmup_proportion = warmup_proportion
        self.adam_epsilon = adam_epsilon

    def train(self, train_dataloader, eval_dataloader, output_dir, save_best=False, eval_metric='f1'):
        """Training loop for bert fine-tuning."""

        t_total = len(train_dataloader) * self.train_epochs
        warmup_steps = len(train_dataloader) * self.warmup_proportion
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        train_iterator = trange(int(self.train_epochs), desc="Epoch")
        #model = self.model
        self.model.to(self.device)
        tr_loss_track = []
        eval_metric_track = []
        output_filename = os.path.join(output_dir, 'pytorch_model.bin')
        metric = float('-inf')

        for _ in train_iterator:
            self.model.train()
            self.model.zero_grad()
            tr_loss = 0
            nr_batches = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                tr_loss = 0
                input_ids, input_mask, labels = batch
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=input_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                scheduler.step()
                tr_loss += loss.item()
                nr_batches += 1
                self.model.zero_grad()

            print("Evaluating the model on the evaluation split...")
            metrics = self.evaluate(eval_dataloader)
            eval_metric_track.append(metrics)
            if save_best:
                if metric < metrics[eval_metric]:
                    self.model.save_pretrained(output_dir)
                    torch.save(self.model.state_dict(), output_filename)
                    print("The new value of " + eval_metric + " score of " + str(metrics[eval_metric]) + " is higher then the old value of " +
                          str(metric) + ".")
                    print("Saving the new model...")
                    metric = metrics[eval_metric]
                else:
                    print(
                        "The new value of " + eval_metric + " score of " + str(metrics[eval_metric]) + " is not higher then the old value of " +
                        str(metric) + ".")

            tr_loss = tr_loss / nr_batches
            tr_loss_track.append(tr_loss)

        if not save_best:
            self.model.save_pretrained(output_dir)
            # tokenizer.save_pretrained(output_dir)
            torch.save(self.model.state_dict(), output_filename)

        return tr_loss_track, eval_metric_track

    def evaluate(self, eval_dataloader):
        """Evaluation of trained checkpoint."""
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        true_labels = []
        data_iterator = tqdm(eval_dataloader, desc="Iteration")
        for step, batch in enumerate(data_iterator):
            input_ids, input_mask, labels = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)

            # loss is only output when labels are provided as input to the model ... real smooth
            logits = outputs[0]
            print(type(logits))
            logits = logits.to('cpu').numpy()
            label_ids = labels.to('cpu').numpy()

            for label, logit in zip(label_ids, logits):
                true_labels.append(label)
                predictions.append(np.argmax(logit))

        # print(predictions)
        # print(true_labels)
        metrics = self.get_metrics(true_labels, predictions)
        return metrics

    def predict(self, predict_dataloader, return_probabilities=False):
        """Testing of trained checkpoint."""
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        probabilities = []
        # true_labels = []
        data_iterator = tqdm(predict_dataloader, desc="Iteration")
        softmax = torch.nn.Softmax(dim=-1)
        for step, batch in enumerate(data_iterator):
            input_ids, input_mask = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)

            # loss is only output when labels are provided as input to the model ... real smooth
            logits = outputs[0]
            #print(type(logits))
            probs = softmax(logits)
            logits = logits.to('cpu').numpy()
            probs = probs.to('cpu').numpy()
            # label_ids = labels.to('cpu').numpy()

            for l, prob in zip(logits, probs):
                # true_labels.append(label)
                predictions.append(np.argmax(l))
                probabilities.append(prob)

        # print(predictions)
        # print(true_labels)
        # metrics = get_metrics(true_labels, predictions)
        if return_probabilities == False:
            return predictions
        else:
            return predictions, probabilities

    def get_metrics(self, true, predicted):
        metrics = {'accuracy': accuracy_score(true, predicted),
                   'recall': recall_score(true, predicted, average="macro"),
                   'precision': precision_score(true, predicted, average="macro"),
                   'f1': f1_score(true, predicted, average="macro")}

        return metrics
