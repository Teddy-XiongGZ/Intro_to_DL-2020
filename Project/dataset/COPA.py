"""
COPA dataset
https://people.ict.usc.edu/~gordon/copa.html
"""

import torch
from torch.utils.data import Dataset, DataLoader
from xml.dom.minidom import parse
import xml.dom.minidom

class COPADataset(Dataset):
  """
  COPA dataset in the following format:
  self.data[index][0(premise)/1(casual)/2(no causal)] returns a string.
  """
  def __init__(self):
    self.data = list()
    copa_corpus = xml.dom.minidom.parse("./datasets/COPA.xml").documentElement
    items = copa_corpus.getElementsByTagName("item")
    for item in items:
      # index = item.getAttribute("id") # unnecessary, for they are always 1~1000
      index = item.getAttribute("most-plausible-alternative")
      p = item.getElementsByTagName("p")[0].childNodes[0].data
      a1 = item.getElementsByTagName("a1")[0].childNodes[0].data
      a2 = item.getElementsByTagName("a2")[0].childNodes[0].data
      entry = list()
      entry.append(p)
      if index == "1":
        entry.append(a1)
        entry.append(a2)
      elif index == "2":
        entry.append(a2)
        entry.append(a1)
      self.data.append(entry)
  def __len__(self):
    return 1000
  def __getitem__(self, index):
    return self.data[index]
