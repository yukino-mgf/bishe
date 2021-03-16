# coding: utf-8
#测试文件，对原始数据进行解码处理
import csv
from urllib.parse import unquote
import re
with open('./data/xssed.csv','r') as f:
    csv_iter = csv.reader(f)
    datas = [data[0] for data in csv_iter]

if datas[-1] == '':
    datas.pop(-1)

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
# i = 0
# for index,data in enumerate(result):
#     if re.search('%[0-7][0-9a-fA-F]',data):
#         i+=1
#         print('---------------')
#         print(data)
#         print(datas[index])
# print('*************',i)
i = 0
for index,data in enumerate(result):
    if re.search('',data):
        i+=1
        print('---------------')
        print(data)
        print(datas[index])
print('*************',i)
        

with open("processed_data.txt",'w') as f:
    for data in result:
        f.write(data+'\n')
