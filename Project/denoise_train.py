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
k = 10
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

      output = model(data, data_len).cpu()

      _, pred = torch.max(output, 1)
      if predictions == None:
        predictions = pred
      else:
        predictions = torch.cat((predictions, pred))
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
class OOF(Dataset):   # One out of five
  """
  COPA dataset utility that prepares necessary data and files.  
  https://people.ict.usc.edu/~gordon/copa.html
  """

  def __init__(self, proportions=(0.5, 0, 0.5), sep_token=" "):
    super().__init__(name="OOF")
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

  def process(self,sample_new):
    posi = []
    nega = []
    # indice = []
    # out = []
    for item in sample_new:
      if item['label'] == 1:
        posi.append(item)
      else:
        nega.append(item)
    times = ((k-1)*len(posi)) // len(nega)
    if ((k-1)*len(posi)) > (times*len(nega)):
      times = times + 1
    tmp = nega * times
    random.shuffle(posi)
    random.shuffle(tmp)
    for i, item in enumerate(posi):
      rank = random.randint(0,k-1)
      tmp = tmp[:rank+i*k] + [item] + tmp[rank+i*k:]
      # indice.append(rank)
    # for i, rank in enumerate(indice):
      # out.append({'texts':tmp[i*k:(i+1)*k],'index':rank})
    # return out
    return tmp

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

      ret.append(self.process(sample_new))
    return (ret[0], ret[1], ret[2])

conf = Config('Denoise', batch_size=k, test_batch_size=k, learning_rate=lr, max_epoch=30, class_num=k) # 5e-6 if 2 gpus
model = Detector(conf, pretrain="roberta-large").to(device)

TEXT = torchtext.data.Field(batch_first=True, use_vocab=False, tokenize=model.tokenize, preprocessing = model.tokenizer.convert_tokens_to_ids, 
                  init_token = model.tokenizer.cls_token_id, eos_token = model.tokenizer.sep_token_id, pad_token = model.tokenizer.pad_token_id, unk_token = model.tokenizer.unk_token_id, 
                  include_lengths = True) # length can be used to minimize the influence of padding. Currently not used.


LABEL = torchtext.data.Field(sequential=False, dtype=torch.long)
# keys in these fields must match the keys in the .jsonl file
fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

# dat = []
# dat.append(OOF((0.4, 0.1, 0.5)))

# manager = DatasetManager(config=conf, device=device, fields=fields)
# manager.add(CombinedDataset("train", dat, test=False))
# manager.add(CombinedDataset("test", dat, train=False, val=False))

dat = OOF((0.8,0.1,0.1))
train_data, val_data, test_data = torchtext.data.TabularDataset.splits(
        path=".",
        train=dat.get_train(),
        validation=dat.get_val(),
        test=dat.get_test(),
        format='json',
        fields=fields
    )

fields["label"][1].build_vocab(train_data)
iters = {}
iters["train"] = torchtext.data.BucketIterator(train_data, batch_size=k, sort_key=lambda x: len(x.text), shuffle=False, device=device)
iters["val"] = torchtext.data.BucketIterator(val_data, batch_size=k, sort_key=lambda x: len(x.text), shuffle=False, device=device)
# test_data, batch_size=self.config.test_batch_size, sort_key=lambda x: len(x.text), shuffle=False, device=self.device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=conf.learning_rate, weight_decay=0.01)
print("=" * 70)
if not iters.get("train") == None: # if this dataset contains train data
  train_losses, val_accs = conf.train(model, optimizer, criterion, iters["train"], iters["val"])
  plot_loss(train_losses)
  plot_acc(val_accs)
if not iters.get("test") == None: # if this dataset contains test data
  model.load_state_dict(torch.load("./checkpoint/" + conf.model_name))
  test_acc, test_loss = conf.test(model, iters["test"], criterion)
  print('-' * 70)
  # print("Test Accuracy: {:.4f}".format(test_acc))
  print('-' * 70)
