#coding: utf-8
import csv,pickle,time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import GeneSeg
from gensim.models.word2vec import Word2Vec
from collections import Counter
learning_rate=0.01
vocabulary_size=3000
batch_size=128
embedding_size=128
num_skips=4
skip_window=5
num_sampled=64
num_iter=5
plot_only=100
# log_dir="word2vec.log"
plt_dir="./file/word2vec.png"
vec_dir="./file/word2vec.pickle"

start=time.time()
words = []
datas = []
#分词 目前仅使用了正样本（xss）
with open('./data/xssed.csv','r',encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        word=GeneSeg(row[0])
        datas.append(word)
        words+=word
"""
datas = [['word0',''word1','word2'...]
         ['word3',''word4','word5'...]    
         ...                          ]
words = ['word0','word1','word2','word3'...]
"""

#将分词结果中的部分词替换为UNK
def build_dataset(datas,words):
    count = [["UNK",-1]]
    counter = Counter(words)
    count.extend(counter.most_common(vocabulary_size-1))
    vocabulary = [c[0] for c in count]
    data_set = []
    for data in datas:
        d_set = []
        for word in data:
            if word in vocabulary:
                d_set.append(word)
            else:
                d_set.append("UNK")
                count[0][1]+=1
        data_set.append(d_set)
    return data_set
"""
data_set = [['word0',''UNK','UNK'...]
            ['word3',''word4','word5'...]    
            ...                            ]
"""

data_set = build_dataset(datas,words)
# 模型训练
model = Word2Vec(data_set,size=embedding_size,\
    window=skip_window,negative=num_sampled,iter=num_iter)
embeddings=model.wv

#绘图
def plot_with_labels(low_dim_embs,labels,filename=plt_dir):
    plt.figure(figsize=(10,10))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),xytext=(5,2),
                     textcoords="offset points",
                     ha="right",
                     va="bottom")
        f_text="vocabulary_size=%d;batch_size=%d;embedding_size=%d;skip_window=%d;num_iter=%d"%(
            vocabulary_size,batch_size,embedding_size,skip_window,num_iter
        )
        plt.figtext(0.03,0.03,f_text,color="green",fontsize=10)
    # plt.show()
    plt.savefig(filename)
tsne=TSNE(perplexity=30,n_components=2,init="pca",n_iter=5000) # TSNE数据降维函数
plot_words=embeddings.index2word[:plot_only]
plot_embeddings=[]
for word in plot_words:
    plot_embeddings.append(embeddings[word])
low_dim_embs=tsne.fit_transform(plot_embeddings)
plot_with_labels(low_dim_embs,plot_words)

#存储
def save(embeddings):
    dic = dict([(embeddings.index2word[i],i)for i in range(len(embeddings.index2word))])
    reverse_dic=dict(zip(dic.values(),dic.keys()))
    word2vec={"dictionary":dic,"embeddings":embeddings,"reverse_dictionary":reverse_dic}
    with open(vec_dir,"wb") as f:
        pickle.dump(word2vec,f)
save(embeddings)
end = time.time()
print("------------------------------------")
print("word2vec embedding training finished")
print("total time: ",end-start)
print("save wordvec to: ",vec_dir)