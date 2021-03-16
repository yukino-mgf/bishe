import torch
import torch.nn as nn
import torch.nn.functional as F
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