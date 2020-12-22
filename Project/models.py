import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer
from transformers import MT5EncoderModel
from transformers import RobertaTokenizer
from transformers import RobertaModel
from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaModel

class Moderl(nn.Module):
  """
  Base class for models used in this project.
  """
  
  def __init__(self, class_num, tokenizer, backbone, hidden_size, interim_dim = 3):
    super(Moderl, self).__init__()
    self.class_num = class_num
    self.tokenizer = tokenizer
    self.backbone = backbone
    self.hidden_size = hidden_size
    self.fc_in = nn.Linear(self.hidden_size, interim_dim)
    self.fc_out = nn.Linear(interim_dim if self.class_num == 1 else self.hidden_size, self.class_num)

  def forward(self, x, x_len):
    if self.class_num == 1:
      return self._forward_BCE(x, x_len)
    else:
      return self._forward_CE(x, x_len)

  def _forward_CE(self, x, x_len):
    x = self.backbone(x).last_hidden_state[:, 0, :]
    return self.fc_out(x)
  
  def _forward_BCE(self, x, x_len):
    x = self.backbone(x).last_hidden_state[:, 0, :]
    x = self.fc_in(x)
    x = torch.sigmoid(x)
    return self.fc_out(x).squeeze(1)

  def tokenize(self, string):
    tokens = self.tokenizer.tokenize(string)
    # reserve space for BOS and EOS
    return tokens[:self.tokenizer.model_max_length - 2]



class Roberta(Moderl):
  def __init__(self, conf, pretrain="roberta-large"):
    tokenizer = RobertaTokenizer.from_pretrained(pretrain)
    backbone = RobertaModel.from_pretrained(pretrain, return_dict=True)
    hidden_size = backbone.config.to_dict()["hidden_size"]
    super(Roberta, self).__init__(class_num=conf.class_num, tokenizer=tokenizer, backbone=backbone, hidden_size=hidden_size)



class XLMRoberta(Moderl):
  def __init__(self, conf, pretrain="xlm-roberta-large"):
    tokenizer = XLMRobertaTokenizer.from_pretrained(pretrain)
    backbone = XLMRobertaModel.from_pretrained(pretrain, return_dict=True)
    hidden_size = backbone.config.to_dict()["hidden_size"]
    super(XLMRoberta, self).__init__(class_num=conf.class_num, tokenizer=tokenizer, backbone=backbone, hidden_size=hidden_size)
      


class MT5(Moderl):
  def __init__(self, conf, pretrain="google/mt5-large"):
    class_num = conf.class_num
    tokenizer = T5Tokenizer.from_pretrained(pretrain)
    backbone = MT5EncoderModel.from_pretrained(pretrain, return_dict=True)
    hidden_size = backbone.config.to_dict()["d_model"]
    super(XLMRoberta, self).__init__(class_num=conf.class_num, tokenizer=tokenizer, backbone=backbone, hidden_size=hidden_size)
