import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import os
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

class CleanBasicRNN(nn.Module):
    def __init__(self, batchSize, nInputs, nNeurons):
        super(CleanBasicRNN, self).__init__()
        
        self.batchSize = batchSize
        self.nNeurons = nNeurons
        
        self.rnn = nn.RNNCell(nInputs, nNeurons)
        self.hx = None
        self.FC = nn.Linear(nNeurons, 6)
        
    def init_layer(self, batch_size=None, device=None):
        bs = batch_size if batch_size is not None else self.batchSize
        dev = device if device is not None else torch.device("cpu")
        self.hx = torch.zeros(bs, self.nNeurons, device=dev)
        
    def forward(self, X):
        if self.hx is None or self.hx.size(0) != X.size(0) or self.hx.device != X.device:
            self.init_layer(X.size(0), X.device)
        self.hx = self.rnn(X, self.hx.detach())
        out = self.FC(self.hx)
        return out

class RecursiveNeuralNetwork():
  def __init__(self,nInputs,nNeurons,epochs):
    
    self.n_neurons = nNeurons
    self.n_inputs = nInputs
    self.lr = 0.01
    self.batch_size = 1000
    self.epochs = epochs

    self.net = CleanBasicRNN(self.batch_size,self.n_inputs,self.n_neurons)
  
  def train(self,trainLoader,testLoader):

    acc_list = list()
    test_list = list()
    loss_list = list()

    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Model instance
    model = self.net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=self.lr)

    for epoch in range(self.epochs):
        train_acc = 0.0
        train_running_loss = 0.0
        model.train()
        print("Epoch Value: ",epoch)

        total_batches = len(trainLoader)
        for batch_idx, netData in enumerate(trainLoader):
            optimizer.zero_grad()
            data = netData[:,0:4].float().to(device)
            targ =  netData[:,4].long().to(device)
            model.init_layer(data.size(0), device)

            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, targ)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            mba = self.get_accuracy(outputs, targ)
            train_acc += mba
            
        model.eval()
        avg_loss = train_running_loss / max(1, total_batches)
        avg_acc = train_acc / max(1, total_batches)
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, avg_loss, avg_acc))

        test_acc,_,_ = self.predict(testLoader)
        test_list.append(test_acc)
        acc_list.append(avg_acc)
        loss_list.append(avg_loss)
    return acc_list,test_list,loss_list

  def get_accuracy(self,logit,target):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / target.size(0)
    return accuracy.item()

  def predict(self,test_loader):
    correct = 0
    total = 0
    device = next(self.net.parameters()).device
    model = self.net
    all_targets = []
    all_preds = []

    model.eval()
    with torch.no_grad():
      for batch_idx, netData in enumerate(test_loader):
        data = netData[:,0:4].float().to(device)
        targ = netData[:,4].long().to(device)
        model.init_layer(data.size(0), device)
        net_out = model(data)
        _, predicted = torch.max(net_out.data, 1)
        total += targ.size(0)
        correct += (predicted == targ).sum().item()
        all_targets.extend(targ.cpu().numpy().tolist())
        all_preds.extend(predicted.cpu().numpy().tolist())
    accuracy = (100. * correct / total) if total > 0 else 0.0
    return accuracy, np.array(all_targets), np.array(all_preds)