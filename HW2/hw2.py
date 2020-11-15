# -*- coding: utf-8 -*-
import argparse
import torch
from torch import nn
import torch.optim as optim
from load_data import TINDataset
from config import Config
from model import OurModel
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--train_loc', type = str, default = './TinyImageNet/train/')
parser.add_argument('--val_loc', type = str, default = './TinyImageNet/val/')
parser.add_argument('--val2_loc', type = str, default = './TinyImageNet-A/')
parser.add_argument('--test_loc', type = str, default = './TinyImageNet/test/')
parser.add_argument('--learning_rate', type = float, default = 5e-3)
parser.add_argument('--weight_decay', type = float, default = 0)
parser.add_argument('--used_model', type = str, default = None)
parser.add_argument('--model_name', type = str, default = 'Temp')
parser.add_argument('--optimizer', type = str, default = 'SGD')
parser.add_argument('--data_path',type = str, default = 'data/64/same/new')
parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--test_batch_size', type = int, default = 50)

args = parser.parse_known_args()[0]

conf = Config(args.model_name)
conf.set_learning_rate(args.learning_rate)
conf.set_batch_size(args.batch_size)
conf.set_test_batch_size(args.test_batch_size)
model = OurModel(args.used_model)
model.to(device)

dataset = TINDataset(args)
print('Finish Loading!')
# np.save(os.path.join(args.data_path,'train_data.npy'), dataset.train_data.numpy())
# np.save(os.path.join(args.data_path,'val_data.npy'), dataset.val_data.numpy())
# np.save(os.path.join(args.data_path,'val2_data.npy'), dataset.val2_data.numpy())
# np.save(os.path.join(args.data_path,'test_data.npy'), dataset.test_data.numpy())
# np.save(os.path.join(args.data_path,'train_label.npy'), dataset.train_label.numpy())
# np.save(os.path.join(args.data_path,'val_label.npy'), dataset.val_label.numpy())
# np.save(os.path.join(args.data_path,'val2_label.npy'), dataset.val2_label.numpy())
# print('Finish Saving!')

# 炼丹炉 The Alchemy
optimizers = {
	'SGD': optim.SGD,
	'Adagrad': optim.Adagrad,
	'Adadelta': optim.Adadelta,
	'RMSprop': optim.RMSprop,
	'Adam': optim.Adam
}
criterion = nn.CrossEntropyLoss()
optimizer = optimizers[args.optimizer](params = model.parameters(), lr=conf.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40)

# losses = conf.train(model, optimizer, scheduler, criterion, dataset)
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

# srun -G 1 --pty --nodelist=thunlp-4 python hw2.py --used_model vit_base_patch16_224 --model_name vit_base_patch16_224 --data_path 'data/64/same/new' --weight_decay 0.0 --batch_size 50
