import os
import time
import torch

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
        data = data.to(self.device)
        target = (batch.label - 1).to(self.device)  # label ranges from 1 to class_num

        output = model(data, data_len)
        if self.class_num == 1:
          loss = criterion(output, target.float())
        else:
          loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*data.size(0)

        if self.class_num == 1:
          pred = output
          train_correct += torch.sum(pred.round()
                                        == target.squeeze(-1))
        else:
          pred = torch.argmax(output, dim=-1)
          train_correct += torch.sum(pred == target.squeeze(-1))

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
      data = data.to(self.device)
      target = (batch.label - 1).to(self.device)
      output = model(data, data_len)

      if self.class_num == 1:
        pred = output
      else:
        pred = torch.argmax(output, dim=-1)

      target = target.squeeze(-1)
      total += len(pred)

      if self.class_num == 1:
        correct += int(torch.sum(pred.round() == target))
        loss = criterion(output, target.float())
      else:
        correct += int(torch.sum(pred == target))
        loss = criterion(output, target)
      test_loss += loss.item()*data.size(0)

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
      data = data.to(self.device)
      output = model(data, data_len).cpu()
      _, pred = torch.max(output, 1)
      if predictions == None:
        predictions = pred
      else:
        predictions = torch.cat((predictions, pred))

    return predictions
