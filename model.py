# model.py
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import timm
class OurModel(nn.Module):
  def __init__(self, model_name):
    super(OurModel, self).__init__()
    #self.efficientnet = EfficientNet.from_pretrained(model_name)
    self.efficientnet = timm.create_model(model_name, pretrained=True)
    self.linear = nn.Linear(1000,100)
  def forward(self,input):
    return self.linear(self.efficientnet(input))
