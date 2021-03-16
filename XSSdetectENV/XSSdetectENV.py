# -*- coding: utf-8 -*-

# 输入动作 action，
# reward = XSSNET(modified XSS)
import pickle,csv,random,torch

from modifier import Xss_modifier
from utils import GeneSeg
from XSSdetector import XSSCNN

raw_data_dir = './data/xssed.csv'
w2v_model_dir = './file/word2vec.pickle'
xss_detector_dir = './file/torch/xsscnn 2021-03-11 15-07-25.pth'
class XSSdetectENV():
    def __init__(self):
        # self.action_space 动作空间 可执行的动作数
        self.action_space = len(Xss_modifier.ACTION_TABLE)
        # self.rawdatas 原始xss字符数据--列表
        with open(raw_data_dir,'r') as f:
            csv_iter = csv.reader(f)
            self.rawdatas = [s[0] for s in csv_iter]
        
        # self.word2vec self.embeddings self.dictionary self.reverse_dictionary 处理 字符 to embeddings
        # word2vec["train_size"]=train_size
        # word2vec["test_size"]=test_size
        # word2vec["input_num"]=input_num 字符串标准长度
        # word2vec["dims_num"]=dims_num 字符串embedding维度
        with open(w2v_model_dir,'rb') as f:
            self.word2vec = pickle.load(f)
        self.embeddings = self.word2vec["embeddings"] # embedding['word'] = embedding vector
        self.reverse_dictionary = self.word2vec["reverse_dictionary"] # number/index--->'word'
        self.dictionary = self.word2vec["dictionary"] # 'word'--->number/index

        self.modifier = Xss_modifier() # 修饰模型的实现，修饰str以绕过检测
        self.detector = XSSCNN(self.word2vec['input_num'],self.word2vec['dims_num']) # CNN检测模型，在step()中检测str是否为xss
        self.detector.load_state_dict(torch.load(xss_detector_dir))
        self.detector.eval()
        # ##########
        # # debug
        # a = torch.randn(1,self.word2vec['dims_num'],self.word2vec['input_num'])
        # print(self.detector(a))
        # ##########
        self.current_sample = '' # 当前样本
        self.enhanced_datas = [] # 保存强化成功的数据

    def step(self, action_number):
        # return observation,reward,down,{}
        is_escape = False
        # step 1: modifier
        self.current_sample = self.modifier.modify(self.current_sample,action_number)
        # step 2: string --> word 分词, 统一长度
        word_list = GeneSeg(self.current_sample)
        if len(word_list)>= self.word2vec["input_num"]:
            word_list = word_list[0:self.word2vec["input_num"]] # python 取左不取右
        else:
            word_list = word_list+["UNK"]*(self.word2vec["input_num"]-len(word_list))
        # step 3: word --> vec 词到向量的映射
        vec_list = []
        for word in word_list:
            if word in self.dictionary.keys():
                vec_list.append(self.embeddings[word])
            else:
                vec_list.append(self.embeddings["UNK"])
        #step 4: vec输入到神经网络，判断输出
        net_input = torch.tensor([vec_list],dtype=torch.float).permute(0,2,1) #将vec_list 扩展到3维，并输出
        result = self.detector(net_input)
        print("detect result:", result)
        reward = 1 - result[0][1].item() # 未逃避成功，则reward=1-检测器判定为xss的概率
        result = torch.argmax(result,dim=-1).item() # result 为[[a,b]]向量，取其中大值的位置，确认是否逃避成功
         
        if result == 0: # 逃避成功
            is_escape=True
            print("escape succeed: "+self.current_sample)
            self.enhanced_datas.append(self.current_sample)
            reward = 5
        return net_input, reward, is_escape, {} # 输入到detector的张量即为环境返回的观测(observation)

    def reset(self):
        # get a new xss sample
        self.current_sample = random.choice(self.rawdatas)
        print("************************************")
        print("current sample: ",self.current_sample)
        ##################################################
        # step 2: string --> word 分词, 统一长度
        word_list = GeneSeg(self.current_sample)
        if len(word_list)>= self.word2vec["input_num"]:
            word_list = word_list[0:self.word2vec["input_num"]] # python 取左不取右
        else:
            word_list = word_list+["UNK"]*(self.word2vec["input_num"]-len(word_list))
        # step 3: word --> vec 词到向量的映射
        vec_list = []
        for word in word_list:
            if word in self.dictionary.keys():
                vec_list.append(self.embeddings[word])
            else:
                vec_list.append(self.embeddings["UNK"])
        #step 4: vec输入到神经网络，判断输出
        net_input = torch.tensor([vec_list],dtype=torch.float).permute(0,2,1) #将vec_list 扩展到3维，并输出
        result = self.detector(net_input)
        print("initial detect result:", result)
        print("*************************************")
        ##################################################

    def render(self):
        pass


if __name__ == '__main__':
    env = XSSdetectENV()
    env.reset()
    for i in range(50):
        _,_,is_escape,_ = env.step(i%env.action_space)
        if is_escape:
            print("total steps:", i+1)
            break

