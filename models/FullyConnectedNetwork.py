import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time

class Net(nn.Module):
    def __init__(self, inputs):
        
        self.inputs = inputs
        
        super(Net, self).__init__()
        self.fc1 = nn.Linear(self.inputs, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc4(x)           
        return F.log_softmax(x,1)

class FullyConnectedNetwork():
    def __init__(self, epoch, inputs):
      
       self.epochs = epoch
       self.net = Net(inputs)
        
    def train(self,train_loader,test_loader):
        
        net = self.net
        batch_size=256
        learning_rate=0.001
        log_interval=1000
        trainAccuracy = []
        testAccuracy = []
      
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        
        criterion = nn.NLLLoss()
    
        for epoch in range(self.epochs):
          correct = 0
          total = 0
          for batch_idx, sub_data in enumerate(train_loader):
            data = sub_data[:,0:4].float()
            targ = sub_data[:,4].float()
            data, targ = Variable(data), Variable(targ)
            # data = data.view(-1, 3*32*32)
            optimizer.zero_grad()
            net_out = net(data.float())
            # print(net_out.shape,targ.shape)
            targ = targ.type(torch.LongTensor)
            loss = criterion(net_out, targ)
            _, predicted = torch.max(net_out.data, 1)
            total += targ.size(0)
            correct += (predicted == targ).sum().item()

            if ((batch_idx % log_interval) == 0):
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data))
            loss.backward()
            optimizer.step()
          print('Test Accuracy: '+str(correct/total))
          trainAcc = correct/total
          trainAccuracy.append(trainAcc)
          _,_,testAcc = self.predict(test_loader)
          testAccuracy.append(testAcc)
        return trainAccuracy,testAccuracy
    
    def predict(self,test_loader):
        
        test_loss = 0
        correct = 0
        total = 0
        net=self.net
        yPred = []
        yTrue = []
        
        for batch_idx, sub_data in enumerate(test_loader):
          data = sub_data[:,0:4].float()
          targ = sub_data[:,4].float()
          #data = data.view(-1, 3*32*32)
          net_out = net(data.float())
          # sum up batch loss
          _, predicted = torch.max(net_out.data, 1)
          total += targ.size(0)
          correct += (predicted == targ).sum().item()
          yPred.append(predicted[0].item())
          yTrue.append(targ[0].item())
    
    
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f},\033[1m \033[4m Accuracy:\033[0m {}/{} ({:.0f}%)\n'.format(test_loss, correct, total,100. * correct / total))
        return yTrue,yPred,(correct/total)