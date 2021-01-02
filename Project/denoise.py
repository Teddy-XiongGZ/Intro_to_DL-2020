import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import jsonlines
import torchtext
import random
import xml.dom.minidom
from transformers import RobertaTokenizer
from transformers import RobertaModel
from transformers import RobertaConfig
from command import download
# from models import Roberta
from plot import plot_loss, plot_acc

lr = 1e-6
k = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config(object):
  def __init__(self, model_name, batch_size, test_batch_size, learning_rate, max_epoch, class_num=2):
    self.batch_size = batch_size
    self.test_batch_size = test_batch_size
    self.max_epoch = max_epoch
    self.class_num = class_num
    self.learning_rate = learning_rate
    self.save_path = './checkpoint'
    self.log_path = './log'
    if not os.path.exists(self.log_path):
      os.mkdir(self.log_path)
    if not os.path.exists(self.save_path):
      os.mkdir(self.save_path)
    self.model_name = model_name
    self.device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu")
  def logging(self, s, print_=True, log_=True):
    if print_:
      print(s)
    if log_:
      with open(os.path.join(self.log_path, self.model_name), 'a+') as f_log:
        f_log.write(s + '\n')
  def train(self, model, optimizer, criterion, train_iter, val_iter):
    model.train()
    self.logging("Training Started, using " +
                  str(criterion) + " and " + str(optimizer) + "\n\n")
    train_losses = []
    val_accs = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(self.max_epoch):
      train_loss = 0.0
      start_time = time.time()
      train_total = train_correct = 0
      for batch in train_iter:
        optimizer.zero_grad()
        data, data_len = batch.text
        target = batch.label - 1  # label ranges from 1 to class_num
        target = (target == 1).nonzero().squeeze(0)
        output = model(data, data_len)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.argmax(output, dim=-1)
        train_correct += torch.sum(pred == target).item()
        train_total += len(pred)
      train_loss = train_loss / train_total
      train_losses.append(train_loss)
      if train_total > 0:
        acc = train_correct / train_total
      else:
        acc = 0.0
      self.logging('[Epoch {:d}] Training Loss: {:.6f}; Time: {:.2f}min; Accuracy: {:.5f}'.format(
        epoch + 1, train_loss, (time.time()-start_time)/60, acc))
      if not val_iter == None:
        self.logging('-' * 70)
        eval_start_time = time.time()
        val_acc, val_loss = self.test(model, val_iter, criterion)
        val_accs.append(val_acc)
        self.logging('Validation Loss: {:.6f}; Time: {:.2f}min; Accuracy: {:.5f}'.format(
            val_loss, (time.time()-eval_start_time)/60, val_acc))
        if val_acc > best_acc:
          best_acc = val_acc
          best_epoch = epoch + 1
          torch.save(model.state_dict(), os.path.join(
                self.save_path, self.model_name))
        model.train()
        self.logging('-' * 70)
    self.logging('Training finished!')
    self.logging('Best epoch: {:d} | Accuracy: {:.5f}'.format(best_epoch, best_acc))
    return train_losses, val_accs
  def test(self, model, test_iter, criterion):
    model.eval()
    total = correct = 0
    test_loss = 0.0
    for batch in test_iter:
      data, data_len = batch.text
      target = batch.label - 1
      target = (target == 1).nonzero().squeeze(0)
      output = model(data, data_len)
      pred = torch.argmax(output, dim=-1)
      total += len(pred)
      correct += int(torch.sum(pred == target))
      loss = criterion(output, target)
      test_loss += loss.item()
    if total > 0:
      accuracy = correct/total
    else:
      accuracy = 0.0
    test_loss = test_loss/total
    return accuracy, test_loss
  def predict(self, model, test_iter):
    model.eval()  # prep model for *evaluation*
    predictions = None
    for batch in test_iter:
      data, data_len = batch.text
      output = model(data, data_len).view(-1).tolist()
      pred = output
      if predictions == None:
        predictions = pred
      else:
        predictions = predictions + pred
    return predictions

class Detector(nn.Module):
  def __init__(self, conf, pretrain = 'roberta-large'):
    super(Detector, self).__init__()
    self.class_num = conf.class_num
    self.tokenizer = RobertaTokenizer.from_pretrained(pretrain)
    self.backbone = RobertaModel.from_pretrained('roberta-large', return_dict=True)
    # self.backbone = RobertaModel(RobertaConfig())
    self.hidden_size = self.backbone.config.to_dict()["hidden_size"]
    self.fc_out = nn.Linear(self.hidden_size, 1)
  def forward(self, x, x_len):
    x = self.backbone(x).last_hidden_state[:, 0, :]
    return self.fc_out(x).permute(1,0)
  def tokenize(self, string):
    tokens = self.tokenizer.tokenize(string)
    # reserve space for BOS and EOS
    return tokens[:self.tokenizer.model_max_length - 2]

from dataset import Dataset, DatasetManager, CombinedDataset
import jsonlines

class SocialIQA_posi():
    """
    SocialIQA dataset utility that prepares necessary data and files.    
    https://leaderboard.allenai.org/socialiqa/submissions/get-started    
    Note: This dataset does not distinguish between cause/effect.
    """
    def __init__(self, proportions=(1.0, .0, .0), sep_token=" ",name="socialiqa_posi"):
        self.name = name
        self.sep_token = sep_token
        self.generate()
    def get_data(self):
        return os.path.join('./download', "{}.jsonl".format(self.name))
    def generate(self):
        """ generate .jsonl files from SocialIQA manually """
        download("https://node0.static.jsonx.ml/socialiqa/socialiqa.jsonl")
        download("https://node0.static.jsonx.ml/socialiqa/socialiqa_label.txt")
        socialiqa_corpus = jsonlines.open("./download/socialiqa.jsonl", mode='r') # unlabelled data
        socialiqa_label = open("./download/socialiqa_label.txt", mode="r") # label
        samples = []
        for item in socialiqa_corpus.iter():
            label = int(socialiqa_label.readline().strip())
            samples.append({"text": item, "label": label})
        jsonlines.open(self.get_data(),'w').write_all(self.postprocess(samples))
    def postprocess(self, sample_list):
        sample_new = []
        for item in sample_list:
            label = item["label"]    # 1 / 2 / 3
            item = item["text"]
            question = item["question"]    # expressed in natural language
            premise = item["context"]
            answer1 = item["answerA"]
            answer2 = item["answerB"]
            answer3 = item["answerC"]
            correct = answer1 if label == 1 else (answer2 if label == 2 else answer3)
            sample_new.append({"text": premise + " " + question + self.sep_token + correct, "label": 1})
        return sample_new

conf = Config('Denoise', batch_size=k, test_batch_size=k, learning_rate=lr, max_epoch=30, class_num=k) # 5e-6 if 2 gpus
model = Detector(conf, pretrain="roberta-large").to(device)
model.load_state_dict(torch.load('checkpoint/Denoise'))

TEXT = torchtext.data.Field(batch_first=True, use_vocab=False, tokenize=model.tokenize, preprocessing = model.tokenizer.convert_tokens_to_ids, 
                  init_token = model.tokenizer.cls_token_id, eos_token = model.tokenizer.sep_token_id, pad_token = model.tokenizer.pad_token_id, unk_token = model.tokenizer.unk_token_id, 
                  include_lengths = True) # length can be used to minimize the influence of padding. Currently not used.


LABEL = torchtext.data.Field(sequential=False, dtype=torch.long)
# keys in these fields must match the keys in the .jsonl file
fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

dat = SocialIQA_posi()

socialiqa = torchtext.data.TabularDataset(dat.get_data(),format='json',fields=fields)

fields["label"][1].build_vocab(socialiqa)
iters = torchtext.data.BucketIterator(socialiqa, batch_size=k, sort_key=lambda x: len(x.text), shuffle=False, device=device)

res = conf.predict(model,iters)

import numpy as np
import pandas as pd
np.save('iqa_score.npy',res)
res = np.load('iqa_score.npy')
res = pd.Series(res)
content = pd.read_table('download/socialiqa.jsonl',header = None,names=['content'])
label = pd.read_table('download/socialiqa_label.txt',header = None,names=['label'])
content['score'] = res
content['label'] = label
content = content.sort_values(by='score',ascending=False)
content = content.iloc[:10000]
jsonlines.open('download/socialiqa.jsonl','w').write_all([eval(x) for x in content['content'].tolist()])
jsonlines.open('download/socialiqa_label.txt','w').write_all(content['label'].tolist())
