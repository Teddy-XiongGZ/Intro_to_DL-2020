"""
Datasets used for the task
Motive:  
- Datasets in NLP are in diverse formats. For instance, 
  COPA uses .xml, while XCOPA and SocialIQA use .jsonl.  
- The labels in different datasets are not always the same. 
  A "premise" label in one dataset might be "context" in
  another one.
- Bigger space for customization.
Attention:
- We assume that the train and development sets provided
  by datasets are of the same distribution.
"""

import os
import random
import jsonlines
import torchtext
import xml.dom.minidom
from command import download

class Dataset():
  """ 
  Base class for the datasets used in the task.  
  The common procedures for dataset generation are:  
  Constructor __init__() called a member function generate(), 
  inside of which the dataset is parsed from its original representation
  to a list of items. Then, split() is called to separate the 
  list into train, validation and test sets. postprocess() is called
  if postprocessing is necessary (e.g. data formatting and augmentation), 
  before executing a write() so that the files could be used for 
  torchtext.data.TabularDataset.
  """
  path = "./download"
  if not os.path.exists(path):
    os.mkdir(path)

  def __init__(self, name="dataset"):
    self.name = name
    self.size = [0, 0, 0]

  def get_train(self):
    return os.path.join(self.path, "{}_train.jsonl".format(self.name))

  def get_val(self):
    return os.path.join(self.path, "{}_val.jsonl".format(self.name))

  def get_test(self):
    return os.path.join(self.path, "{}_test.jsonl".format(self.name))

  def get_size(self):
    return self.size

  def split(self, samples, proportions=(1.0, .0, .0)):
    """
    args:  
      samples(list): the list of samples to be split
      proportions(tuple): corresponds respectively to the proportion of train, val and test data.
    TODO: we assume that "samples" is a list, but we can modify this function to accept any iterable. 
    """
    if not (sum(proportions) == 1):
      raise Exception("proportions of train, val and test do not sum to 1!")
    # random.shuffle(samples) # shuffle when making batch iters
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
      self.size[0] += 1

    writer = jsonlines.open(self.get_val(), mode='w')
    for item in samples[1]:
      writer.write(item)  # in val or test, but not in test
      self.size[1] += 1

    writer = jsonlines.open(self.get_test(), mode='w')
    for item in samples[2]:
      writer.write(item)  # in test
      self.size[2] += 1



class CombinedDataset(Dataset):
  """ 
  This class is used for combining several datasets together.
  Args:
    name: name of this combined dataset.
    datasets: an iterable containing the datasets to be combined.
    train/val/test: whether the result dataset contains train, val and test. 
      For instance, if train=False, train data in individual datasets will not be used.
  """

  def __init__(self, name, datasets, train=True, val=True, test=True):
    internal_name = name + ":"
    for dataset in datasets:
      internal_name += " " + dataset.name
    super().__init__(name=internal_name)
    self.datasets = datasets
    self.train = train
    self.val = val
    self.test = test
    self.generate()

  def generate(self): # TODO: this function can be optimized by direct file-IO rather than jsonlines
    samples_train = []
    samples_val = []
    samples_test = []
    for dataset in self.datasets:
      if os.path.exists(dataset.get_train()) and self.train:
        corpus = jsonlines.open(dataset.get_train(), mode='r')
        for item in corpus.iter():
          samples_train.append(item)

      if os.path.exists(dataset.get_val()) and self.val:
        corpus = jsonlines.open(dataset.get_val(), mode='r')
        for item in corpus.iter():
          samples_val.append(item)

      if os.path.exists(dataset.get_test()) and self.test:
        corpus = jsonlines.open(dataset.get_test(), mode='r')
        for item in corpus.iter():
          samples_test.append(item)

    self.write((samples_train, samples_val, samples_test))


class COPA(Dataset):
  """
  COPA dataset utility that prepares necessary data and files.  
  https://people.ict.usc.edu/~gordon/copa.html
  """

  def __init__(self, proportions=(0.5, 0, 0.5), sep_token=" "):
    super().__init__(name="copa")
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

        sample_new.append({"text": premise + self.sep_token +
                        correct, "label": 1, "tag": tag})
        sample_new.append({"text": premise + self.sep_token +
                        wrong, "label": 0, "tag": tag})
        if not sample_list == samples[2]:  # test set should not be augmented
          # one item can be augmented into four samples: cause/effect; correct/wrong
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
  available_languages = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]

  def __init__(self, proportions=(.0, 1/6, 5/6), lang="zh", sep_token=" "):
    super().__init__(name="xcopa-" + lang)
    if not (lang in self.available_languages):
      raise Exception("The specified language is not supported in XCOPA!")
    self.proportions = proportions
    self.sep_token = sep_token
    self.generate(lang)

  def generate(self, lang):
    """ generate .jsonl files from XCOPA manually """
    download("https://node0.static.jsonx.ml/xcopa/{}.jsonl".format(lang))

    samples = []
    corpus = jsonlines.open("./download/{}.jsonl".format(lang), mode='r')
    for item in corpus.iter():
      samples.append(item)

    self.write(self.postprocess(self.split(samples, self.proportions)))

  def postprocess(self, samples):
    ret = []
    for sample_list in samples:
      sample_new = []
      for item in sample_list:
        label = item["label"]  # 0 / 1
        tag = item["question"]  # cause / effect
        premise = item["premise"]
        answer1 = item["choice1"]
        answer2 = item["choice2"]

        correct = answer1 if label == 0 else answer2
        wrong = answer2 if label == 0 else answer1

        sample_new.append({"text": premise + self.sep_token +
                        correct, "label": 1, "tag": tag})
        sample_new.append({"text": premise + self.sep_token +
                        wrong, "label": 0, "tag": tag})
        if not sample_list == samples[2]:  # test set should not be augmented
          # one item can be augmented into four samples: cause/effect; correct/wrong
          sample_new.append({"text": correct + self.sep_token + premise,
                          "label": 1, "tag": "cause" if tag == "effect" else "effect"})
          sample_new.append({"text": wrong + self.sep_token + premise, "label": 0,
                          "tag": "cause" if tag == "effect" else "effect"})
      ret.append(sample_new)

    return (ret[0], ret[1], ret[2])



class WinoGrande(Dataset):
  """
  WinoGrande dataset utility that prepares necessary data and files.  
  https://winogrande.allenai.org/  
  Note: sep_token is not available in this dataset.
  """

  def __init__(self, proportions=(1.0, .0, .0)):
    super().__init__(name="winogrande")
    self.proportions = proportions
    self.generate()

  def generate(self):
    """ generate .jsonl files from WinoGrande manually """
    download("https://node0.static.jsonx.ml/winogrande/winogrande.jsonl")

    winogrande_corpus = jsonlines.open(
        "./download/winogrande.jsonl", mode='r')
    samples = []
    for item in winogrande_corpus.iter():
      samples.append(item)

    self.write(self.postprocess(self.split(samples, self.proportions)))

  def postprocess(self, samples):
    ret = []
    for sample_list in samples:
      sample_new = []
      for item in sample_list:
        label = int(item["answer"])  # 1 / 2
        premise = item["sentence"]
        answer1 = item["option1"]
        answer2 = item["option2"]

        correct = answer1 if label == 1 else answer2
        wrong = answer2 if label == 1 else answer1
        
        sample_new.append({"text": premise.replace("_", correct), "label": 1})
        sample_new.append({"text": premise.replace("_", wrong), "label": 0})
      ret.append(sample_new)

    return (ret[0], ret[1], ret[2])



class SocialIQA(Dataset):
  """
  SocialIQA dataset utility that prepares necessary data and files.  
  https://leaderboard.allenai.org/socialiqa/submissions/get-started  
  Note: This dataset does not distinguish between cause/effect.
  """

  def __init__(self, proportions=(1.0, .0, .0), sep_token=" ", denoise=True):
    super().__init__(name="socialiqa_denoise" if denoise else "socialiqa")
    self.proportions = proportions
    self.sep_token = sep_token
    self.denoise = denoise
    self.generate()

  def generate(self):
    """ generate .jsonl files from SocialIQA manually """
    denoise_dir = "_denoise" if self.denoise else ""
    download("https://node0.static.jsonx.ml/socialiqa/socialiqa{}.jsonl".format(denoise_dir))
    download("https://node0.static.jsonx.ml/socialiqa/socialiqa_label{}.txt".format(denoise_dir))

    socialiqa_corpus = jsonlines.open("./download/socialiqa{}.jsonl".format(denoise_dir), mode='r') # unlabelled data
    socialiqa_label = open("./download/socialiqa_label{}.txt".format(denoise_dir), mode="r") # label
    samples = []
    for item in socialiqa_corpus.iter():
      label = int(socialiqa_label.readline().strip())
      samples.append({"text": item, "label": label})

    self.write(self.postprocess(self.split(samples, self.proportions)))

  def postprocess(self, samples):
    ret = []
    for sample_list in samples:
      sample_new = []
      for item in sample_list:
        label = item["label"]  # 1 / 2 / 3
        item = item["text"]
        question = item["question"]  # expressed in natural language
        premise = item["context"]
        answer1 = item["answerA"]
        answer2 = item["answerB"]
        answer3 = item["answerC"]

        correct = answer1 if label == 1 else (answer2 if label == 2 else answer3)
        wrong1 = answer2 if label == 1 else (answer1)
        wrong2 = answer2 if label == 3 else (answer3)

        sample_new.append({"text": premise + " " + question + self.sep_token +
                           correct, "label": 1})
        sample_new.append({"text": premise + " " + question + self.sep_token +
                           wrong1, "label": 0})
        sample_new.append({"text": premise + " " + question + self.sep_token +
                           wrong2, "label": 0})
        if not sample_list == samples[2]:  # test set should not be augmented
          sample_new.append({"text": correct + self.sep_token + premise,
                            "label": 1})
          sample_new.append({"text": wrong1 + self.sep_token + premise,
                            "label": 0})
          sample_new.append({"text": wrong2 + self.sep_token + premise,
                            "label": 0})
      ret.append(sample_new)

    return (ret[0], ret[1], ret[2])



class DatasetManager:
  """
  An utility that manages the datasets used for the entire pretrain-tune-test lifecycle.  
  """
  _datasets = []
  _index = 0
  _vocab = False

  def __init__(self, config=None, device=None, fields=None):
    self.fields = fields
    self.config = config
    self.device = device

  def add(self, dataset):
    """
    Add a dataset to the end of workflow.  
    It will work even after next() has been called.  
    """
    self._datasets.append(dataset)

  def reset(self):
    """
    Reset the workflow. next() now returns the first dataset.
    """
    self._index = 0

  def next(self, split=True, iter=True):
    """
    Return the next dataset iterators if split and iter, dataset splits if split, or the dataset itself if both are False.
    It always returns None if there are no more datasets.  
    Return (if split or iter):
      a dict containing "train", "val" or "test" iters/data depending on whether the input dataset contains them.
    """
    if self._index >= len(self._datasets):
      return None
    dataset = self._datasets[self._index]
    self._index += 1
    if not split:
      return dataset
    
    train_data, val_data, test_data = torchtext.data.TabularDataset.splits(
        path=".",
        train=dataset.get_train(),
        validation=dataset.get_val(),
        test=dataset.get_test(),
        format='json',
        fields=self.fields
    )

    if not self._vocab:
      self.fields["label"][1].build_vocab(train_data)
      self._vocab = True

    if not iter:
      data = {}
      data["name"] = dataset.name
      if dataset.size[0] > 0:
        data["train"] = train_data
      if dataset.size[1] > 0:
        data["val"] = val_data
      if dataset.size[2] > 0:
        data["test"] = test_data
      return data

    iter = {}
    iter["name"] = dataset.name
    if dataset.size[0] > 0:
      iter["train"] = torchtext.data.BucketIterator(
        train_data, batch_size=self.config.batch_size, shuffle=True, sort_key=lambda x: len(x.text), device=self.device)
    if dataset.size[1] > 0:
      iter["val"] = torchtext.data.BucketIterator(
        val_data, batch_size=self.config.test_batch_size, sort_key=lambda x: len(x.text), device=self.device)
    if dataset.size[2] > 0:
      iter["test"] = torchtext.data.BucketIterator(
        test_data, batch_size=self.config.test_batch_size, sort_key=lambda x: len(x.text), device=self.device)
    return iter
