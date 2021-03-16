import csv,re
from urllib.parse import unquote
#############################
with open('./data/normal_examples-20w.csv','r') as f:
    csv_iter = csv.reader(f)
    ndatas = [data[0] for data in csv_iter]

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

ndatas = decoder(ndatas)

i = [0,0]
invalid_char_count=0
container = []
for index,data in enumerate(ndatas):
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
            flag = 1
            invalid_char_count+=1
            break
    if flag == 0:
        container.append(data)
    # elif len(data)<=1000:
    #     remove.append(data)
print("find %[][]:",i[0])
print("find &#:",i[1])
print('invalid char sentences:',invalid_char_count)

with open('normal-data-decode-20w.txt','w') as f:
    for data in container:
        f.write(data+'\n')