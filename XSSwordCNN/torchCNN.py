import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

import pickle
import os
import json
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# result = tensor.permute(dim0,dim1,dim2) 多维矩阵的转置
class xssdataset(Dataset):
    def __init__(self,path,embeddings,reverse_dictionary):
        super(xssdataset).__init__()
        with open(path,'r') as f:
            datas = f.read().split('\n')
        datas.remove('')
        self.embeddings = embeddings
        self.reverse_dictionary = reverse_dictionary
        self.texts = []
        self.labels = []
        for row in datas:
            text,label = row.split('|')
            self.texts.append(json.loads(text))
            self.labels.append(json.loads(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        data = self.texts[index]
        label = self.labels[index]
        data_embed = []
        for d in data:
            if d != -1:
                data_embed.append(self.embeddings[self.reverse_dictionary[d]])
            else:
                data_embed.append([0.0] * len(self.embeddings["UNK"]))
        #data_embed size (text length, embedding size)--->(embedding size, text length)
        return (torch.tensor(data_embed,dtype=torch.float).permute(1,0), np.argmax(label))


# CNN 的输入向量为（batch size, embedding size, text length)
class XSSCNN(torch.nn.Module):
    def __init__(self,text_length,embedding_size): # text_length : 输入语句的词汇数，embedding_size : 词汇的embedding大小
        super(XSSCNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embedding_size,out_channels=64,kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=32,kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.2)
        self.final_length = int((int((text_length - 2)/2) - 2)/2)
        self.dense1 = nn.Linear(32*self.final_length,100)
        self.dense2 = nn.Linear(100,2)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.drop2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.dense2(self.dense1(x))
        # out = F.softmax()
        return F.softmax(x,dim=1)

# text_length=30
# embedding_size=64
# mod = XSSCNN(30,64)
# a = torch.randn(20,64,30)
# b = mod(a)
# print(b.shape)

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


w2v_dict_dir = './file/word2vec.pickle'


if __name__ == '__main__':
    # 读取w2v文件内的字典
    with open(w2v_dict_dir, "rb") as f:
        word2vec = pickle.load(f)
    embeddings = word2vec["embeddings"] # embedding['word'] = embedding vector
    reverse_dictionary = word2vec["reverse_dictionary"] #number--->'word'
    # train_size=word2vec["train_size"]
    # test_size=word2vec["test_size"]
    dims_num = word2vec["dims_num"]
    input_num =word2vec["input_num"]
    
    train_set = xssdataset('./file/pre_datas_train.txt',embeddings,reverse_dictionary)
    test_set = xssdataset('./file/pre_datas_test.txt',embeddings,reverse_dictionary)
    batch_size = 500
    train_loader = DataLoader(train_set,batch_size,shuffle=True)
    test_loader = DataLoader(test_set,batch_size,shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = XSSCNN(input_num,dims_num).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    with SummaryWriter(log_dir='./run/exp1') as writer:
        writer.add_graph(net,input_to_model=torch.randn(batch_size,dims_num,input_num).to(device))
    for epoch in range(1,6):
        train(net,device,train_loader,optimizer,epoch)
        test(net,device,test_loader)
    torch.save(net.state_dict(), "./file/torch/xsscnn {}.pth".format(time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())))