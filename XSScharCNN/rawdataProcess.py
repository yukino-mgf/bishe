# coding: utf-8
#测试文件，对原始数据进行解码处理
import csv
from urllib.parse import unquote
import re
import matplotlib.pyplot as plt

def decoder(datas):
    result = []
    for data in datas:
        data = data.replace('<br/>','')
        data = data.replace('<br>','')
        data = unquote(data)
        data = data.replace('<br/>','')
        data = data.replace('<br>','')
        data = unquote(data)
        data = data.replace('<br/>','')
        data = data.replace('<br>','')
        data = unquote(data)
        data = data.replace('\r','')
        data = data.replace('\n','')
        result.append(data)
    return result

def plot_hist(data_list,filename):
    plt.figure()
    plt.hist(data_list,bins=500)
    plt.savefig(filename)
    plt.close()

with open('xss-data-decode-3w.txt','r') as f:
    datas = f.read().split('\n')

i = [0,0,0]
container = []
for index,data in enumerate(datas):
    flag = 0
    if re.search('%[0-7][0-9a-fA-F]',data):
        i[0]+=1
        print('---------------')
        print(data)
        flag = 1
    if re.search('&#',data):
        i[1]+=1
        print('---------------')
        print(data)
        flag = 1
    for char in data:
        if ord(char)>=128:
            i[2]+=1
            flag = 1
            print('---------------')
            print(data)
            break
    if flag == 0:
        container.append(data)

with open('xss-data-decode-3w.txt','w') as f:
    for data in container:
        f.write(data+'\n')