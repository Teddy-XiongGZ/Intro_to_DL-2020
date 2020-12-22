import torch.nn.functional as F
import torch.nn as nn
import torch

from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaModel

class XLMRoberta(nn.Module):
  def __init__(self, conf, pretrain="xlm-roberta-large"):
    super(Moderl, self).__init__()
    self.class_num = conf.class_num
    self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    self.backbone = XLMRobertaModel.from_pretrained(
        'xlm-roberta-large', return_dict=True)
    self.hidden_size = self.roberta.config.to_dict()["hidden_size"]
    self.fc_out = nn.Linear(self.hidden_size, self.class_num)

  def forward(self, x, x_len):
    return self.fc_out(self.backbone(x).last_hidden_state[:, 0, :])

  def tokenize(string):
    tokens = self.tokenizer.tokenize(string)
    # reserve space for BOS and EOS
    tokens = tokens[:self.tokenizer.model_max_length - 2]
    return tokens
