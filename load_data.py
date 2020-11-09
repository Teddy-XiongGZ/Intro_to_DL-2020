# load_data.py
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class TINDataset(torch.utils.data.Dataset):
  """ TinyImageNet Dataset """
  TRAIN = 0
  VALIDATE = 1
  TEST = 2
  _status = TRAIN # mark the current state of the dataset
  
  def __init__(self, args):
    self.train_loc = args.train_loc
    self.val_loc = args.val_loc
    self.test_loc = args.test_loc
    # 对输入矩阵进行转置、转化为tensor、中心化
    #trans = transforms.Compose([transforms.ToTensor()])
    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (118.57,115.17,98.72),std=(68.10,65.77,69.41))])
    #trans = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    print("----------Loading training data----------")
    print('train_loc:',self.train_loc)
    self.train_data = torch.zeros(100 * 1000, 3, 64, 64)
    for i in range(100):
      for j in range(1000):
        filename = self.train_loc+str(i)+'/'+str(i)+'_'+str(j)+'.jpg'
        self.train_data[i * 1000 + j] = trans(mpimg.imread(filename).astype(float))
    print('size:',self.train_data.shape)
    print("--------------Finish loading--------------")

    print("----------Loading validation data----------")
    print('validation_loc:',self.val_loc)
    self.val_data = torch.zeros(100 * 100, 3, 64, 64)
    for i in range(100):
      for j in range(100):
        filename = self.val_loc+str(i)+'/'+str(i)+'_10'+str(j).rjust(2,'0')+'.jpg'
        self.val_data[i * 100 + j] = trans(mpimg.imread(filename).astype(float))
    print('size:',self.val_data.shape)
    print("--------------Finish loading--------------")
    
    print("----------Loading test data----------")
    print('test_loc:',self.test_loc)
    self.test_data = torch.zeros(10000, 3, 64, 64)
    for i in range(10000):
        filename = self.test_loc+str(i)+'.jpg'
        self.test_data[i] = trans(mpimg.imread(filename).astype(float))
    print('size:',self.test_data.shape)
    print("--------------Finish loading--------------")

    self.train_label = torch.mul(torch.LongTensor([i for i in range(100)]).view(-1,1), torch.ones(1000).long().view(1,-1)).view(-1)
    self.val_label = torch.mul(torch.LongTensor([i for i in range(100)]).view(-1,1), torch.ones(100).long().view(1,-1)).view(-1)

  def __getitem__(self, index):
    if self._status == self.TRAIN:
      return self.train_data[index], self.train_label[index] # data, label(target)
    elif self._status == self.VALIDATE:
      return self.val_data[index], self.val_label[index]
    else:
      return self.test_data[index]

  def __len__(self):
    if self._status == self.TRAIN:
      return 100000
    elif self._status == self.VALIDATE:
      return 10000
    else:
      return 10000
  
  def train(self):
    self._status = self.TRAIN

  def validate(self):
    self._status = self.VALIDATE

  def test(self):
    # if self._status != self.TEST:
    #   del self.train_data
    #   del self.train_label
    #   del self.val_data
    #   del self.val_label
    self._status = self.TEST
