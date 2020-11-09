# -*- coding: utf-8 -*-
import argparse
import torch
from torch import nn
import torch.optim as optim
from load_data import TINDataset
from config import Config
from model import OurModel
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser()
parser.add_argument('--train_loc', type = str, default = './TinyImageNet/train/')
parser.add_argument('--val_loc', type = str, default = './TinyImageNet/val/')
parser.add_argument('--test_loc', type = str, default = './TinyImageNet/test/')
parser.add_argument('--learning_rate', type = float, default = 5e-3)
parser.add_argument('--weight_decay', type = float, default = 5e-5)
parser.add_argument('--model_name', type = str, default = None)
args = parser.parse_known_args()[0]

conf = Config(args.model_name)
conf.set_learning_rate(args.learning_rate)
model = OurModel(args.model_name)
model.to(device)

dataset = TINDataset(args)

# 炼丹炉 The Alchemy
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(params = model.parameters(), lr=conf.learning_rate, weight_decay=args.weight_decay)

losses = conf.train(model, optimizer, criterion, dataset)

model_path = os.path.join(conf.save_path, conf.model_name)
model.load_state_dict(torch.load(model_path))
conf.test_all(model, criterion, dataset)

dataset.test()
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=conf.test_batch_size)
result = conf.predict(model, test_loader)

file = open("./submit_"+conf.model_name+".csv", mode="w")
file.write("Id,Category\n")
for i in range(len(result)):
  file.write(str(i) + ".jpg," + str(int(result[i])) + "\n")
file.flush()
file.close()
