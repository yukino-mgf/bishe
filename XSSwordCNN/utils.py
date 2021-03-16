#coding: utf-8
import nltk
import re
from urllib.parse import unquote
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as ktf

import csv
def GeneSeg(payload):
    #数字泛化为"0"
    payload=payload.lower()
    payload=unquote(unquote(payload))#两次解码
    payload,num=re.subn(r'\d+',"0",payload)
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)
def init_session():
    gpu_options=tf.GPUOptions(allow_growth=True)
    ktf.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

if __name__ == '__main__':
    with open("./data/xssed.csv",'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        data = []
        for raw in reader:
            if re.search('%[0-9a-eA-E]{2}',unquote(raw[0])):
                data.append([raw[0],unquote(raw[0]),unquote(unquote(raw[0]))])

    i = 1
    print('-----------------------------------------------')
    print(data[i][0])
    print('-----------------------------------------------')
    print(data[i][1])
    print('-----------------------------------------------')
    print(data[i][2])
    print('-----------------------------------------------')
    print(unquote(unquote(data[i][0].replace('<br/>', ''))))