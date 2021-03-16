# from tensorflow.keras.utils import to_categorical
# a = [0]*3+[1]*2
# print(a)
# print(to_categorical(a))

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# a = [1,2,3]
# b = [1,2,3,4,5]
# c = [a,b]
# print(c)
# print(pad_sequences(c,maxlen=4))

string=[]  #有的题目要输出字符串，但是有时候list更好操作，于是可以最后list转string提交
for i in range(0,5):
    string.append('M')              
str1=''.join(string)
print(str1)
print(string)
print(type(string))

import csv

with open("./file/pre_datas_test.csv","r") as f:
    fc = csv.reader(f)
    a = list(fc)
    print(a[0])
    print(type(a[0]))
    s = ''.join(a[0])
    print(s)