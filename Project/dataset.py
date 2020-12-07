import os
import random
import torch
import jsonlines
import xml.dom.minidom
from command import download

class COPA():
  """
  COPA dataset utility that prepares necessary data and files.
  https://people.ict.usc.edu/~gordon/copa.html
  """
  path = "./download"
  train_file = "copa_train.jsonl"
  val_file = "copa_val.jsonl"

  def __init__(self):
    # download preprocessed files
    download("https://node0.static.jsonx.ml/copa/{}".format(self.train_file))
    download("https://node0.static.jsonx.ml/copa/{}".format(self.val_file))

    # we can also generate them ourselves
    # generate_from_xml(tokenizer)

  def get_train(self):
    return os.path.join(self.path, self.train_file)

  def get_val(self):
    return os.path.join(self.path, self.val_file)

  def generate_from_xml(self, tokenizer):
    """ generate .jsonl files from COPA .xml manually """
    download("https://node0.static.jsonx.ml/copa/copa.xml")
    sep_token = tokenizer.sep_token
    copa_corpus = xml.dom.minidom.parse("./download/copa.xml").documentElement
    items = copa_corpus.getElementsByTagName("item")
    samples = []
    for item in items:
        # index = item.getAttribute("id") # unnecessary, for they are always 1~1000
        index = int(item.getAttribute("most-plausible-alternative"))  # 1 / 2
        tag = item.getAttribute("asks-for")  # cause / effect
        premise = item.getElementsByTagName("p")[0].childNodes[0].data
        answer1 = item.getElementsByTagName("a1")[0].childNodes[0].data
        answer2 = item.getElementsByTagName("a2")[0].childNodes[0].data

        correct = answer1 if index == 1 else answer2
        wrong = answer2 if index == 1 else answer1

        # one item can be augmented into four samples: cause/effect; correct/wrong
        samples.append({"text": premise + sep_token +
                        correct, "label": 1, "tag": tag})
        samples.append({"text": premise + sep_token +
                        wrong, "label": 0, "tag": tag})
        samples.append({"text": correct + sep_token + premise,
                        "label": 1, "tag": "cause" if tag == "effect" else "effect"})
        samples.append({"text": wrong + sep_token + premise, "label": 0,
                        "tag": "cause" if tag == "effect" else "effect"})

    samples_val = random.sample(samples, int(0.1 * len(samples)))
    writer = jsonlines.open(os.path.join(self.path, self.train_file), mode='w')
    for item in samples_val:
        writer.write(item)

    writer = jsonlines.open(os.path.join(self.path, self.val_file), mode='w')
    for item in samples:
        if item not in samples_val:
            writer.write(item)
