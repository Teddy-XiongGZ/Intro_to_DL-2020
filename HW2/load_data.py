# custom transform object dealing with ImageNet-A examples
import torch
from torchvision import datasets, transforms

class MyResize(object):
    """Rescale the image in a sample to a given size, and converts it to a tensor

    Args:
      WARNING: tuple as output_size is not implemented.
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, bigger of image edges is matched
            to output_size keeping aspect ratio the same. The result image 
            will have black paddings on shorter edges.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size

        # else:
        #   if (self.output_size[0] > self.output_size[1]):
        #     new_h = new_w = self.output_size[1]
        #   else:
        #     new_h = new_w = self.output_size[0]

        new_h, new_w = int(new_h), int(new_w)
        sample = torch.tensor(sample).permute(2, 0, 1)

        # if len(sample.shape) == 3: # colored picture
        #   sample = sample.permute(2, 0, 1) # (channel, width, height)
        # else:
        #   temp = torch.zeros(3, sample.shape[0], sample.shape[1])
        #   temp[:] = sample;
        #   print(temp.shape)
        #   sample = temp; 
        
        img = transforms.Resize((new_h, new_w))(sample)

        pad_size = abs(new_h - new_w) # makes sure that the result is exact
        if (new_h > new_w):
          img = transforms.Pad((int(pad_size / 2), 0, int((pad_size + 1) / 2), 0))(img)
        else:
          img = transforms.Pad((0, int(pad_size / 2), 0, int((pad_size + 1) / 2)))(img)

        return img

class MyNormalize(object):
    """Normalize a image tensor
    """

    def __call__(self, data): # assumes the same shape as an image tensor
      assert(len(data.shape) == 3)
      # mean = []
      # std = []
      # for i in range(3):
      #   mean.append(torch.mean(data[i]))
      #   std.append(torch.std(data[i]))
      #transforms.Normalize(mean = (118.57,115.17,98.72),std=(68.10,65.77,69.41), inplace=True)(data)
      data = data / 255
      transforms.Normalize(mean = (0.4650,0.4516,0.3871),std=(0.2671,0.2579,0.2722), inplace=True)(data)
      return data

# load_data.py
from torchvision import datasets, transforms
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import numpy as np

class TINDataset(torch.utils.data.Dataset):
  """ TinyImageNet Dataset """
  TRAIN = 0
  VALIDATE = 1
  VALIDATE_2 = 2
  TEST = 9
  _status = TRAIN # mark the current state of the dataset
  
  def __init__(self, args):
    self.train_loc = args.train_loc
    self.val_loc = args.val_loc
    self.val2_loc = args.val2_loc
    self.test_loc = args.test_loc
    self.data_path = args.data_path
    print(self.data_path)


    # moved here so val_label can be updated while loading TinyImageNetA
    # self.train_label = torch.mul(torch.LongTensor([i for i in range(100)]).view(-1,1), torch.ones(1000).long().view(1,-1)).view(-1)
    # self.val_label = torch.mul(torch.LongTensor([i for i in range(100)]).view(-1,1), torch.ones(100).long().view(1,-1)).view(-1)
    # self.val2_label = torch.zeros(5337)

    # 对输入矩阵进行转置、转化为tensor、中心化
    #trans = transforms.Compose([transforms.ToTensor()])
    #trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (118.57,115.17,98.72),std=(68.10,65.77,69.41))])
    trans = transforms.Compose([transforms.ToTensor(), MyNormalize()])
    transA = transforms.Compose([MyResize(64), MyNormalize()])

    self.train_data = torch.from_numpy(np.load(os.path.join(self.data_path,'train_data.npy')))
    self.val_data = torch.from_numpy(np.load(os.path.join(self.data_path,'val_data.npy')))
    self.val2_data = torch.from_numpy(np.load(os.path.join(self.data_path,'val2_data.npy')))
    self.test_data = torch.from_numpy(np.load(os.path.join(self.data_path,'test_data.npy')))
    self.train_label = torch.from_numpy(np.load(os.path.join(self.data_path,'train_label.npy')))
    self.val_label = torch.from_numpy(np.load(os.path.join(self.data_path,'val_label.npy')))
    self.val2_label = torch.from_numpy(np.load(os.path.join(self.data_path,'val2_label.npy')))

    # print("----------Loading validation2 data ----------")
    # print("validation2_loc", self.val2_loc)
    # self.val2_data = torch.zeros(5337, 3, 64, 64) # 一共就5337个有效样本+20个无效样本
    # count = 0
    # for i in range(100):
    #   list = os.listdir(self.val2_loc + str(i)) #列出文件夹下所有的目录与文件
    #   for j in range(0,len(list)):
    #     filename = self.val2_loc + str(i) + "/" + list[j]
    #     img = mpimg.imread(filename).astype(float)
    #     if (len(img.shape) == 3 and img.shape[2] == 3): # 有效
    #       self.val2_data[count] = transA(img)
    #       self.val2_label[count] = i
    #       count += 1
    # print(count)
    # print('size:',self.val2_data.shape)
    # print("--------------Finish loading--------------")

    # print("----------Loading training data----------")
    # print('train_loc:',self.train_loc)
    # self.train_data = torch.zeros(100 * 1000, 3, 64, 64)
    # for i in range(100):
    #   for j in range(1000):
    #     filename = self.train_loc+str(i)+'/'+str(i)+'_'+str(j)+'.jpg'
    #     self.train_data[i * 1000 + j] = trans(mpimg.imread(filename).astype(float))
    # print('size:',self.train_data.shape)
    # print("--------------Finish loading--------------")

    # print("----------Loading validation data----------")
    # print('validation_loc:',self.val_loc)
    # self.val_data = torch.zeros(100 * 100, 3, 64, 64)
    # for i in range(100):
    #   for j in range(100):
    #     filename = self.val_loc+str(i)+'/'+str(i)+'_10'+str(j).rjust(2,'0')+'.jpg'
    #     self.val_data[i * 100 + j] = trans(mpimg.imread(filename).astype(float))
    
    # print('size:',self.val_data.shape)
    # print("--------------Finish loading--------------")
    
    # print("----------Loading test data----------")
    # print('test_loc:',self.test_loc)
    # self.test_data = torch.zeros(10000, 3, 64, 64)
    # for i in range(10000):
    #     filename = self.test_loc+str(i)+'.jpg'
    #     self.test_data[i] = trans(mpimg.imread(filename).astype(float))
    # print('size:',self.test_data.shape)
    # print("--------------Finish loading--------------")

  def __getitem__(self, index):
    if self._status == self.TRAIN:
      return self.train_data[index], self.train_label[index] # data, label(target)
    elif self._status == self.VALIDATE:
      return self.val_data[index], self.val_label[index]
    elif self._status == self.VALIDATE_2:
      return self.val2_data[index], self.val2_label[index]
    else:
      return self.test_data[index]

  def __len__(self):
    if self._status == self.TRAIN:
      return self.train_data.shape[0]
    elif self._status == self.VALIDATE:
      return self.val_data.shape[0]
    elif self._status == self.VALIDATE_2:
      return self.val2_data.shape[0]
    else:
      return self.test_data.shape[0]
  
  def train(self):
    self._status = self.TRAIN

  def validate(self):
    self._status = self.VALIDATE
  def validate2(self):
    self._status = self.VALIDATE_2
  def test(self):
    # if self._status != self.TEST:
    #   del self.train_data
    #   del self.train_label
    #   del self.val_data
    #   del self.val_label
    self._status = self.TEST
