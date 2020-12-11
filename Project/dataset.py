"""
Datasets used for the task
"""

import os
import random
import jsonlines
import xml.dom.minidom
from command import download

class Dataset():
  """ Base class for the datasets used in the task. """
  name = "dataset"
  path = "./download"

  def get_train(self):
    return os.path.join(self.path, "{}_train.jsonl".format(self.name))

  def get_val(self):
    return os.path.join(self.path, "{}_val.jsonl".format(self.name))

  def get_test(self):
    return os.path.join(self.path, "{}_test.jsonl".format(self.name))

  def split(self, samples, proportions=(1.0, .0, .0)):
    """
    args:
      samples(list): the list of samples to be split
      proportions(tuple): corresponds respectively to the proportion of train, val and test data.
    TODO: we assume that "samples" is a list, but we can modify this function to accept any iterable. 
    """
    if not (sum(proportions) == 1):
      raise Exception("proportions of train, val and test do not sum to 1!")
    random.shuffle(samples)
    test_count = int(len(samples) * proportions[2]); 
    val_count = int(len(samples) * proportions[1]);
    train_count = len(samples) - test_count - val_count
    return (samples[:train_count], samples[train_count:train_count + val_count], samples[train_count + val_count:])
    
  def postprocess(self, samples):
    """
    do postprocess here.
    args:
      samples(tuple): (train samples, val samples, test samples), each as a list
    """
    return samples

  def write(self, samples):
    """ 
    write the samples to .jsonl files.
    args:
      samples(tuple): (train samples, val samples, test samples), each as a list
    """
    writer = jsonlines.open(self.get_train(), mode='w')
    for item in samples[0]:
      writer.write(item)

    writer = jsonlines.open(self.get_val(), mode='w')
    for item in samples[1]:
      writer.write(item)  # in val or test, but not in test

    writer = jsonlines.open(self.get_test(), mode='w')
    for item in samples[2]:
      writer.write(item)  # in test



class COPA(Dataset):
  """
  COPA dataset utility that prepares necessary data and files.
  https://people.ict.usc.edu/~gordon/copa.html
  """

  def __init__(self, proportions=(1.0, .0, .0), sep_token="</s>"):
    self.name = "COPA"
    self.proportions = proportions
    self.sep_token = sep_token
    # we can also generate them ourselves
    # this is now the default choice
    self.generate()

  def generate(self):
    """ generate .jsonl files from COPA .xml manually """
    download("https://node0.static.jsonx.ml/copa/copa.xml")

    copa_corpus = xml.dom.minidom.parse("./download/copa.xml").documentElement
    items = copa_corpus.getElementsByTagName("item")
    samples = []
    for item in items:
      samples.append(item) # parsing is delayed until postprocessing

    self.write(self.postprocess(self.split(samples, self.proportions)))

  def postprocess(self, samples):
    ret = []
    for sample_list in samples:
      sample_new = []
      for item in sample_list:
        # index = item.getAttribute("id") # unnecessary, for they are always 1~1000
        label = int(item.getAttribute("most-plausible-alternative"))  # 1 / 2
        tag = item.getAttribute("asks-for")  # cause / effect
        premise = item.getElementsByTagName("p")[0].childNodes[0].data
        answer1 = item.getElementsByTagName("a1")[0].childNodes[0].data
        answer2 = item.getElementsByTagName("a2")[0].childNodes[0].data

        correct = answer1 if label == 1 else answer2
        wrong = answer2 if label == 1 else answer1

        # one item can be augmented into four samples: cause/effect; correct/wrong
        sample_new.append({"text": premise + self.sep_token +
                        correct, "label": 1, "tag": tag})
        sample_new.append({"text": premise + self.sep_token +
                        wrong, "label": 0, "tag": tag})
        sample_new.append({"text": correct + self.sep_token + premise,
                        "label": 1, "tag": "cause" if tag == "effect" else "effect"})
        sample_new.append({"text": wrong + self.sep_token + premise, "label": 0,
                        "tag": "cause" if tag == "effect" else "effect"})
      ret.append(sample_new)
    return (ret[0], ret[1], ret[2])



class XCOPA(Dataset):
  """
  XCOPA dataset utility that prepares necessary data and files.
  https://github.com/cambridgeltl/xcopa
  """

  def __init__(self, proportions=(1.0, .0, .0), sep_token="</s>"):
    self.name = "XCOPA"
    self.proportions = proportions
    self.sep_token = sep_token
    self.generate()

  def generate(self):
    """ generate .jsonl files from XCOPA manually """
    download("https://node0.static.jsonx.ml/xcopa/xcopa.jsonl")

    xcopa_corpus = jsonlines.open("./download/xcopa.jsonl", mode='r')
    samples = []
    for item in xcopa_corpus.iter():
      samples.append(item)

    self.write(self.postprocess(self.split(samples, self.proportions)))

  def postprocess(self, samples):
    ret = []
    for sample_list in samples:
      sample_new = []
      for item in sample_list:
        label = item["label"]  # 1 / 2
        tag = item["question"]  # cause / effect
        premise = item["premise"]
        answer1 = item["choice1"]
        answer2 = item["choice2"]

        correct = answer1 if label == 1 else answer2
        wrong = answer2 if label == 1 else answer1

        # one item can be augmented into four samples: cause/effect; correct/wrong
        sample_new.append({"text": premise + self.sep_token +
                        correct, "label": 1, "tag": tag})
        sample_new.append({"text": premise + self.sep_token +
                        wrong, "label": 0, "tag": tag})
        sample_new.append({"text": correct + self.sep_token + premise,
                        "label": 1, "tag": "cause" if tag == "effect" else "effect"})
        sample_new.append({"text": wrong + self.sep_token + premise, "label": 0,
                        "tag": "cause" if tag == "effect" else "effect"})
      ret.append(sample_new)

    return (ret[0], ret[1], ret[2])
