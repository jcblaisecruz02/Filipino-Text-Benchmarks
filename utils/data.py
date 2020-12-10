import torch
import numpy as np

def str2bool(v):
    if v.lower()  == 'true': return True
    elif v.lower() == 'false': return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def process_labels(labels):
    # If the task has multiple label columns
    if len(labels.shape) == 2:
        return torch.tensor(labels).float()

    # If the task has one label columns
    elif len(labels.shape) == 1:
        if type(labels[0]) is not int:
            labellist = list(set(labels))
            encodings = [labellist.index(l) for l in labels]
            return torch.tensor(encodings)
        return torch.tensor(labels)

def process_data(text, labels, tokenizer, msl=128):
    # Process text
    if len(text.shape) == 2: # Sentence pair tasks
        text = [tuple(t) for t in text]
        encodings = tokenizer(text, truncation='longest_first', padding='max_length', max_length=msl)
    elif len(text.shape) == 1: # Sentence tasks
        encodings = tokenizer(list(text), truncation='longest_first', padding='max_length', max_length=msl)

    # Process labels
    targets = process_labels(labels)

    # Convert to tensordataset
    if 'distilbert' in str(type(tokenizer)) or 'roberta' in str(type(tokenizer)):
        dataset = torch.utils.data.TensorDataset(torch.tensor(encodings.input_ids),
                                                torch.tensor(encodings.attention_mask), 
                                                targets)
    else:
        dataset = torch.utils.data.TensorDataset(torch.tensor(encodings.input_ids), 
                                                torch.tensor(encodings.token_type_ids), 
                                                torch.tensor(encodings.attention_mask), 
                                                targets)

    return dataset