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
        self.hx = torch.randn(batchSize, nNeurons)
        self.FC = nn.Linear(nNeurons, 6)
        
    def init_layer(self):
        
        self.hx = torch.randn(self.batchSize, self.nNeurons)
        
    def forward(self, X):
        self.hx = self.rnn(X, self.hx)
        out = self.FC(self.hx)
        return out

class RecursiveNeuralNetwork():
  def __init__(self,nInputs,nNeurons,epochs):
    
    self.n_neurons = nNeurons
    self.n_inputs = nInputs
    self.lr = 0.1
    self.batch_size = 1000
    self.epochs = epochs

    self.net = CleanBasicRNN(self.batch_size,self.n_inputs,self.n_neurons)
  
  def train(self,trainLoader,testLoader):

    acc_list = list()
    test_list = list()
    loss_list = list()
    trainAccuracy = []
    testAccuracy = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model instance
    model = self.net

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=self.lr)

    for epoch in range(self.epochs):
        train_acc = 0.0
        train_running_loss = 0.0
        model.train()
        print("Epoch Value: ",epoch)
        
        

        for batch_idx, netData in enumerate(trainLoader):
            optimizer.zero_grad()
            data = netData[:,0:4].float()
            targ =  netData[:,4].float()
            model.init_layer()
            

            # forward + backward + optimize
            outputs = model(data)
            targ = targ.type(torch.LongTensor)
            loss = criterion(outputs, targ)
            loss.backward(retain_graph=True)
            optimizer.step()

            train_running_loss += loss.detach().item()
            mba = self.get_accuracy(outputs, targ)
            train_acc += mba
            
        model.eval()
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss / batch_idx, train_acc/batch_idx))

        # test_acc,_,_ = self.predict(testLoader)
        # test_list.append(test_acc)
        # acc_list.append(train_acc/batch_idx)
        # loss_list.append(train_running_loss/batch_idx)
    return acc_list,test_list,loss_list

  def get_accuracy(self,logit,target):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data) .sum()
    accuracy = 100.0 * corrects/self.batch_size
    return accuracy.item()

  def predict(self,test_loader):
    correct = 0
    total = 0
    model = self.net

    for batch_idx, netData in enumerate(test_loader):
      data = netData[:,0:4].float()
      targ = netData[:,4].float()
      net_out = model(data.float())
      _, predicted = torch.max(net_out.data, 1)
      total += targ.size(0)
      correct += (predicted == targ).sum().item()
    return (100. * correct / total),targ,predicted