# config.py
import os
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class Config(object):
    def __init__(self, model_name):
        self.batch_size = 100
        self.test_batch_size = 50
        self.max_epoch = 30
        self.class_num = 100
        self.learning_rate = 0.005
        self.drop_rate = 0.1
        self.test_epoch = 2#4
        self.save_path = './checkpoint'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_learning_rate(self,learning_rate):
            self.learning_rate = learning_rate

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(os.path.join("log", self.model_name)), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model, optimizer, criterion, dataset):
        self.logging("Training Started, using " + str(criterion) + " and " + str(optimizer) + "\n\n")
        losses = []
        best_acc = 0.0
        best_epoch = 0
        model.train()
        dataset.train()
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epoch):
            train_loss = 0.0
            start_time = time.time()
            train_total = train_correct = 0

            for data,target in train_loader:
                data = data.to(self.device)                                             #这样会导致内存泄漏从而爆显存吗？
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data)#得到预测值

                loss = criterion(output,target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()*data.size(0)

                pred = torch.argmax(output, dim=-1).cpu()
                train_total += len(pred)
                train_correct += int(torch.sum(pred == target.squeeze(-1).cpu()))

            train_loss = train_loss / len(train_loader.dataset)
            losses.append(train_loss)
            if train_total > 0:
                acc = train_correct / train_total
            else:
                acc = 0.0
            self.logging('[Epoch {:d}] Training Loss: {:.6f}; Time: {:.2f}min; Acc: {:.4f}'.format(
            epoch + 1, train_loss, (time.time()-start_time)/60, acc))
            if (epoch + 1) % self.test_epoch == 0:
                self.logging('-' * 70)
                eval_start_time = time.time()
                model.eval()
                dataset.validate()
                val_acc = self.test(model, dataset)
                self.logging('Test on val_set\nTime: {:.2f}min; Test Accuracy: {:.4f}'.format((time.time()-eval_start_time)/60,val_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(),os.path.join(self.save_path, self.model_name))
                model.train()
                dataset.train()
                self.logging('-' * 70)
        self.logging('Training finished!')
        self.logging('Best epoch: {:d} | acc: {:.4f}'.format(best_epoch, best_acc))
        return losses
    
    def test(self, model, dataset):
        data_loader = DataLoader(dataset=dataset, batch_size=self.test_batch_size, shuffle=False)
        total = correct = 0
        for data, target in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data).cpu()
                # convert output probabilities to predicted class
                pred = torch.argmax(output, dim=-1)
                # compare predictions to true label
                target = target.squeeze(-1).cpu()
                total += len(pred)
                correct += int(torch.sum(pred == target))
        
        if total > 0:
            accuracy = correct/total
        else:
            accuracy = 0.0
        return accuracy

    # this function actually does the validation process, not the testing: 暂时把验证集当测试集用
    def test_all(self, model, criterion, dataset):
        test_loss = 0.0
        class_correct = list(0. for i in range(self.class_num))
        class_total = list(0. for i in range(self.class_num))
        model.eval() # prep model for *evaluation*

        dataset.validate()
        test_loader = DataLoader(dataset = dataset, batch_size = self.test_batch_size)

        for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # update test loss 
                test_loss += loss.item()*data.size(0)
                # convert output probabilities to predicted class
                #_, pred = torch.max(output, 1)
                pred = torch.argmax(output, dim=-1)
                # compare predictions to true label
                #target = target.data.view_as(pred)
                target = target.squeeze(-1)
                correct = np.squeeze(pred.eq(target))
                # calculate test accuracy for each object class
                for i in range(len(target)):
                        label = target.data[i]
                        class_correct[label] += correct[i].item()
                        class_total[label] += 1

        # calculate and self.logging avg test loss
        test_loss = test_loss/len(test_loader.dataset)
        self.logging('\nTest Loss: {:.6f}\n'.format(test_loss))

        self.logging('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(class_correct) / np.sum(class_total),
                np.sum(class_correct), np.sum(class_total)))

        for i in range(self.class_num):
                if class_total[i] > 0:
                        self.logging('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                                str(i), 100 * class_correct[i] / class_total[i],
                                np.sum(class_correct[i]), np.sum(class_total[i])))
                else:
                        self.logging('Test Accuracy of %5s: N/A (no training examples)' % (str[i]))

    def predict(self, model, test_loader):
        model.eval() # prep model for *evaluation*
        predictions = None

        for data in test_loader:
                data = data.to(self.device)
                output = model(data)
                # convert output probabilities to predicted class
                _, pred = torch.max(output, 1)
                if predictions == None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred))
        return predictions
