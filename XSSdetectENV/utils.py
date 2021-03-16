import nltk,re
from urllib.parse import unquote

def GeneSeg(payload):
    #数字泛化为"0"
    payload=payload.lower()
    payload=unquote(unquote(payload))#两次解码
    payload,num=re.subn(r'\d+',"0",payload)
    #替换url为 http://u
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