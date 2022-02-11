import torch
import utils.tokenization as tokenization
import pandas as pd
import numpy as np
import pickle

# get mask to decide whether data is supervised data or not.
def get_labeled_mask(all_size, labeled_size):
    labeled_mask = torch.zeros((all_size, 1))
    labeled_mask[range(labeled_size)] = 1
    labeled_mask = 0.1 < labeled_mask

    return labeled_mask

class AGProcessor():
    def __init__(self):
        self.unsup_label = '0'

    def get_labels(self):
        return ['0', '1', '2', '3', '4']

    def create_examples(self, input_file, is_unsup=True, aug=False, labeled_examples=[]):
        examples = []

        if aug and is_unsup:
            with open(input_file, 'r') as f:
                arr = pd.read_csv(f, index_col=False, header=None).to_numpy()
                for i in range(len(arr)):
                    text_a = arr[i,0]
                    guid = 1
                    gt = labeled_examples[i]

                    if is_unsup:
                        label=self.unsup_label
                    else:
                        label = gt

                    examples.append(dict(guid=guid, text_a=text_a, text_b=None, label=label, gt=gt))
            return examples

        elif aug and not is_unsup:
            with open(input_file, 'r') as f:
                arr = pd.read_csv(f, index_col=False, header=None).to_numpy()
                for i in range(len(arr)):
                    text_a = arr[i,0]
                    guid = 1
                    gt = str(0)

                    if is_unsup:
                        label=self.unsup_label
                    else:
                        label = gt

                    examples.append(dict(guid=guid, text_a=text_a, text_b=None, label=label, gt=gt))
            return examples
            
        else:
            with open(input_file, 'r') as f:
                arr = pd.read_csv(f, index_col=False, header=None).to_numpy()
                print(arr.shape)
            for i in range(len(arr)):
                guid = i
                text_a = arr[i,-1]
                gt = str(arr[i,0])

                if is_unsup:
                    label=self.unsup_label
                else:
                    label = gt

                examples.append(dict(guid=guid, text_a=text_a, text_b=None, label=label, gt=gt))
            return examples

class YahooProcessor():
    def __init__(self):
        self.unsup_label = '0'

    def get_labels(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    def create_examples(self, input_file, is_unsup=True):
        examples = []

        with open(input_file, 'r') as f:
            arr = pd.read_csv(f, index_col=False, header=None).to_numpy()
        for i in range(len(arr)):
            guid = i
            text_a = arr[i,-1]
            gt = str(arr[i,0])

            if is_unsup:
                label=self.unsup_label
            else:
                label = gt

            examples.append(dict(guid=guid, text_a=text_a, text_b=None, label=label, gt=gt))
        return examples


class DbpediaProcessor():
    def __init__(self):
        self.unsup_label = '0'

    def get_labels(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

    def create_examples(self, input_file, is_unsup=True):
        examples = []

        with open(input_file, 'r') as f:
            arr = pd.read_csv(f, index_col=False, header=None).to_numpy()
        for i in range(len(arr)):
            guid = i
            text_a = arr[i,-1]
            gt = str(arr[i,0])

            if is_unsup:
                label=self.unsup_label
            else:
                label = gt

            examples.append(dict(guid=guid, text_a=text_a, text_b=None, label=label, gt=gt))
        return examples
