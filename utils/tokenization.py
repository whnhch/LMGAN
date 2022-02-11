# # This file comes originally from https://github.com/google-research/bert/blob/master/tokenization.py
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
import torch
import collections
import re
import unicodedata
import six

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def getTokens(tokenizer, examples, processor, MAX_LEN, factor):
    input_ids = []
    attention_masks = []
    labels = []
    segmentation_ids = []
    gts = []

    for example in examples:
      label = example['label']
      text = example['text_a']
      gt = example['gt']

      encoded_dict = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,  # Pad & truncate all sentences.
        padding='max_length',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.,
        truncation=True
        )

      cnt = 1
      if label != processor.unsup_label:
          cnt = factor

      for i in range(cnt):
          input_ids.append(encoded_dict['input_ids'])
          attention_masks.append(encoded_dict['attention_mask'])
          labels.append(processor.get_labels().index(label))
          gts.append(processor.get_labels().index(gt))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    gts = torch.tensor(gts)

    return input_ids, attention_masks, gts
