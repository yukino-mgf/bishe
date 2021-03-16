import random,time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

class char_embed_dataset(Dataset):
    def __init__(self,xss_path=None,normal_path=None):
        super().__init__()
        self.xss_raw_datas = []
        if xss_path:
            with open(xss_path,'r') as f:
                self.xss_raw_datas = f.read().split('\n')
            if '' in self.xss_raw_datas:
                self.xss_raw_datas.remove('')    
        self.xss_number = len(self.xss_raw_datas)
        self.xss_shuffle_list = random.sample(range(self.xss_number),self.xss_number)

        self.normal_raw_datas = []
        if normal_path:
            with open(normal_path,'r') as f:
                self.normal_raw_datas = f.read().split('\n')
            if '' in self.normal_raw_datas:
                self.normal_raw_datas.remove('')
        self.normal_number = len(self.normal_raw_datas)
        self.normal_shuffle_list = random.sample(range(self.normal_number),self.normal_number)
        
        self.dataset = self.xss_raw_datas
        for index in range(self.xss_number):
            self.dataset.append(self.normal_raw_datas[ self.normal_shuffle_list[index] ])
        self.labels = [1]*self.xss_number+[0]*self.xss_number
        self.maxlen = 500

    def __len__(self):
        return 2*self.xss_number

    def __getitem__(self,index):
        data = self.dataset[index]
        label = self.labels[index]
        trans = []
        for char in data:
            trans.append(ord(char))
        data = F.one_hot(torch.tensor(trans),num_classes=128).tolist()
        
        if len(data)<self.maxlen:
            data = data + [[0]*128]*(self.maxlen-len(data))
        else:
            data = data[0:self.maxlen]
        return torch.tensor(data,dtype=torch.float).permute(1,0), label
    
    def divide_dataset(self,p=0.7):
        x = int(p*self.xss_number)
        train_set = []
        test_set = []
        for i in range(x):
            train_set.append(self.xss_raw_datas[self.xss_shuffle_list[i]])
        for i in range(x):
            train_set.append(self.normal_raw_datas[self.normal_shuffle_list[i]])
        train_labels = [1]*x+[0]*x
        for i in range(self.xss_number-x):
            test_set.append(self.xss_raw_datas[self.xss_shuffle_list[x+i]])
        for i in range(self.xss_number-x):
            test_set.append(self.normal_raw_datas[self.normal_shuffle_list[x+i]])
        test_labels = [1]*(self.xss_number-x)+[0]*(self.xss_number-x)
        return train_set,train_labels,test_set,test_labels
    
    def init_from(self,dataset,labels):
        self.dataset = dataset
        self.labels = labels
        self.xss_number = int(len(labels)/2)
        self.normal_number = self.xss_number
        self.xss_raw_datas = dataset[0:self.xss_number]
        self.normal_raw_datas = dataset[self.xss_number:]
        self.xss_shuffle_list = random.sample(range(self.xss_number),self.xss_number)
        self.normal_shuffle_list = random.sample(range(self.normal_number),self.normal_number)

class charCNN(nn.Module):
    def __init__(self,input_length,embedding_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=embedding_size,out_channels=64,kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=32,kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.2)
        self.final_length = int((int((input_length - 2)/2) - 2)/2)
        self.dense1 = nn.Linear(32*self.final_length,100)
        self.dense2 = nn.Linear(100,2)
    def forward(self,x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.maxpool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = F.tanh(x)
        x = self.maxpool2(x)
        x = self.drop2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.dense2(F.tanh(self.dense1(x)))
        # out = F.softmax()
        return F.softmax(x,dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) # output是向量 target是数值
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    base_set = char_embed_dataset('./data/xss-data-decode-3w.txt','./data/normal-data-decode-20w.txt')
    
    trainset = char_embed_dataset()
    testset = char_embed_dataset()
    train_set,train_labels,test_set,test_labels = base_set.divide_dataset()
    trainset.init_from(train_set,train_labels)
    testset.init_from(test_set,test_labels)

    batch_size = 500
    train_loader = DataLoader(trainset,batch_size,shuffle=True)
    test_loader = DataLoader(testset,batch_size,shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = charCNN(500,128).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    with SummaryWriter(log_dir='./run/exp1') as writer:
        writer.add_graph(net,input_to_model=torch.randn(batch_size,128,500).to(device))
    for epoch in range(1,6):
        train(net,device,train_loader,optimizer,epoch)
        test(net,device,test_loader)
    torch.save(net.state_dict(), "./file/torch/charCNN {}.pth".format(time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())))