"""

Author: Andraž Pelicon

"""

from keras_preprocessing.sequence import pad_sequences
import math
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def prepare_data_for_prediction(data, tokenizer, max_len, batch_size):
    """Prepares the input text. Adds CLS and SEP tokens.
    If the text is longer than max length it cuts the first part and the last part of the text and cocatenates them
    together.
    :param data: (list) list of input text sequences
    :param tokenizer: (transformers tokenizer) BERT tokenizer
    :param max_len: (int) maximum length of input sequence
    :param batch_size: (int) batch size
    :return pytorch dataloader
    """
    sentences = ["[CLS] " + sentence for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

    cut_tokenized_sentences = []
    for tokenized_sentence in tokenized_sentences:
        if len(tokenized_sentence) < max_len:
            cut_tokenized_sentences.append(tokenized_sentence + ["[SEP]"])
        elif len(tokenized_sentence) > max_len:
            tokenized_sentence = tokenized_sentence[:math.floor(max_len / 2)] + \
                           tokenized_sentence[-(math.ceil(max_len / 2) - 1):] + ["[SEP]"]
            cut_tokenized_sentences.append(tokenized_sentence)
        else:
            tokenized_sentence = tokenized_sentence[:-1] + ["[SEP]"]
            cut_tokenized_sentences.append(tokenized_sentence)

    #print("Example of tokenized sentence:")
    #print(cut_tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in cut_tokenized_sentences]
    #print("Printing encoded sentences:")
    #print(input_ids[0])

    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks)
    sampler = SequentialSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader