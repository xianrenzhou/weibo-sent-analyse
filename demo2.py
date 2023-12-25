from distutils.command.config import config
import time
from certifi import contents
from sklearn import datasets

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
#from importlib import import_module
from datetime import timedelta
from tqdm import tqdm
import pickle as pkl
import os
import re
from sklearn.preprocessing import LabelEncoder
import random
from tkinter import *
from tkinter import messagebox
import spider as sp
import dataanaly as da
MAX_VOCAB_SIZE = 10000  # 词表长度限制
import csv
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):#random
        
        # self.model_name = 'TextCNN'
        #self.train_path = '/datasets/5fbcdfa05005208e83d1ede4-momodel/news'                                # 训练集
        self.train_path = '/datasets/weibo_senti_100k'             
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = 'mydata/data/vocab3.pkl'                                # 词表
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 128
        self.embedding_pretrained = None
        self.n_vocab = 0
        self.embed =300
        self.filter_sizes = (1,2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)    
        self.dropout = 0.4  
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class3.txt', encoding='utf-8').readlines()]
        self.num_classes = len(self.class_list)    
        self.device = torch.device('cpu')   # 设备 
        self.learning_rate = 1e-3                                       # 学习率
        self.num_epochs = 30                                            # epoch数
        self.save_path = dataset + '/saved_dict/' + 'cnn6' +''+ '.ckpt'
        self.log_path = dataset + '/log/' + 'cnn'
        self.require_improvement = 1000 
        self.cfggb = 0
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        print((config.n_vocab, config.embed, config.n_vocab - 1))
        print(self.embedding)
            #
            #input()
        self.convs = nn.ModuleList(#layer = nn.ModuleList([nn.Conv2d(in_channels=128, out_channels=64,kernel_size])
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
            
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        z = conv(x)
        # print(z)
        # print(z.size())
        # input()
        x = F.relu(conv(x)).squeeze(3)
        # print("eelu之后")
        # print(x)
        # print(x.size())
        # input()
        #x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        # print("   ---")
        # print(x.size())
        # input()
        return x

    def forward(self, x):


        # print("新tensor")
        out = self.embedding(x[0])
   
        out = out.unsqueeze(1)

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # print("zxhfouis")
        # print(out.size())
        # input()
        out = self.dropout(out)
        out = self.fc(out)
        return out

def reprocfun(content):
    reObj = re.compile("@.*?(?=：| |:)")
    reObj2 = re.compile("<.*?(?=>)")
    x=reObj.sub("",content)
    x = reObj2.sub("",x)
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    text = re.sub(pattern,'',x)
    # text = text.replace("啊","")s
    # text = text.replace("呢","")
    # text = text.replace("啦","")
    # text = text.replace("的","")
    # text = text.replace("是","")
    # text = text.replace("呀","")
    # text = text.replace("了","")
    # text = text.replace("和","")
    # text = text.replace("又","")
    return text

def build_vocab(file_path, tokenizer, max_size, min_freq):#    tokenizer = lambda x: [y for y in x]  
    vocab_dic = {}
    this_dataset = {}
    dataset_path = "E:/workspace/datamining/test/MYETEST/grA/datasets/weibo_senti_100k"
    files= os.listdir(dataset_path)
    for file in files:
        path = os.path.join(dataset_path, file)
        if not os.path.isdir(path) and not file[0] == '.': # 跳过隐藏文件和文件夹
            f = open(path, 'r',  encoding='UTF-8') # 打开文件
            for line in f.readlines():
                 this_dataset[line] = file[:-4]
    #types = ("科技", "社会", "娱乐", "财经", "体育")
    # types = ("正向","负向")
    types = ("愤怒","害怕","惊讶","开心","难过","喜欢","厌恶")
    for k, v in list(this_dataset.items()):
        # print(k, "Type:", v, '\n')
            # break
        lin = k.strip()
        if not lin:
            continue
        content = lin
        # pattern = re.compile(r'[^\u4e00-\u9fa5]')
        # # print(content.replace("~!@#$%^&*()_+`}{|\[\]\:\";\-\\\='<>?,./，,!。、][《》？；：‘“{【】}|、！@#￥%……&*（）——+=-",""))
        # print(content)
        # tem = re.sub(pattern,'',content)
        # print(tem)

        # input()
        content = reprocfun(content)

        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}#enum编号
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})    
    return vocab_dic
def build_dataset(config):
    tokenizer = lambda x: [y for y in x]
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        with open( 'mydata/data/vocab3.pkl', 'wb') as f:
            pkl.dump(vocab, f, pkl.HIGHEST_PROTOCOL)
        # pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")


    pad_size = 32 
    def ld_fun(data_list):
        contents = []
        for line in data_list:
            #lin = line.strip()

            try:
                content,label = line.split('\t')#content 为前面的新闻，label为后面标签
                # content = content.replace(",，。. ","")
                content = reprocfun(content)
            except:
                print(line)
                print("dfshiou")
                input()
            words_line = []
            token = tokenizer(content)#tokenizer = lambda x: [y for y in x]，拆成单字
            seq_len = len(token)
            if pad_size:#短补长截,短的字符长度不变，长的截断为pad_size
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len))
        return contents
    sentences = [] 
    target = [] # 类别
    dataset_path = 'datasets/weibo_senti_100k'
    # labels = {'负向': '0', '正向': '1'}
   # labels = {'负向': '0', '正向': '1'}
    labels = { '喜欢': '0','厌恶':'1','开心':'2','难过':'3','愤怒':'4','惊讶':'5','害怕':'6'}
    files = os.listdir(dataset_path)
    for file in files:
        path = os.path.join(dataset_path, file)
        if not os.path.isdir(path) and not file[0] == '.':
            with open(path, 'r', encoding='UTF-8') as f: # 打开文件
                for line in f.readlines():
                    content = line
                    content = content.replace('\t','')
                    content = reprocfun(content)
                    content = content+'\t'
                    tags = labels[file[:-4]]
                    content = content+tags
                    # print(content)
                    sentences.append(content)
                    

    random.shuffle(sentences)
    offset = (int)(len(sentences)*0.8)
    print(offset)
    train_pre = sentences[:offset]
    val_pre = sentences[offset:]
    train = ld_fun(train_pre)
    val = ld_fun(val_pre)
    return vocab,train,val       

class DatasetIterater(object):#迭代器
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        
        self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def get_time_dif(start_time):#已经使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter      


def train(config, model, train_iter, dev_iter):


    # test(config, model, test_iter)
    # return 0

    # model.load_state_dict(torch.load(config.save_path))
    # model.eval()
    # for a,b in test_iter:
    #     outputs = model(a)
    #     x = torch.max(outputs.data, 1)[1].cpu().numpy()
    #     print(x)
    ##batch_size置1，test路径修改
    #return 0
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))


        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)

            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path,_use_new_zipfile_serialization=False)
                    improve = '增长'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)

            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            if test == True:
                print(predic)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def predict(model,text,cfggb):



    def ld_fun(data_list):
        tokenizer = lambda x: [y for y in x]
        pad_size = 32
        vocab_path= 'mydata/data/vocab3.pkl'
        vocab = pkl.load(open(vocab_path, 'rb'))
        contents = []
        
        for line in data_list:
            #lin = line.strip()
            try:
                content,label = line.split('\t')#content 为前面的新闻，label为后面标签
            except:
                print(line)
                print("dfshiou")
                input()
            words_line = []
            token = tokenizer(content)#tokenizer = lambda x: [y for y in x]，拆成单字
            seq_len = len(token)
            if pad_size:#短补长截,短的字符长度不变，长的截断为pad_size
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len))
        return contents    
    text = text.replace('\t','')
    
    content = text
    content = content+'\t'
    content = content+'0'
    sentences = []
    sentences.append(content)

    a = 'mydata'
    b = '123'
    config2 = Config(a,b)
    config2.n_vocab = cfggb

    mpd = ld_fun(sentences)
    mpd_iter = build_iterator(mpd, config2)
    
    # input()
    #labels = {0: '负向', 1: '正向'}
    labels = { 0:'like',1:'disgust',2:'happiness',3:'sadness',4:'anger',5:'surprise',6:'fear'}
    # labels = { 0:'正向',1:'负向',2:'正向',3:'负向',4:'负向',5:'负向',6:'负向'}
    model.eval()
    with torch.no_grad():
        for a,b in mpd_iter:
            outputs = model(a)
            x = torch.max(outputs.data, 1)[1].cpu().numpy()
            # y = torch.max(outputs.data, 1)[0].cpu().numpy()
            outputs.data[0][x[0]] = -999
            y = torch.max(outputs.data, 1)[1].cpu().numpy()
            print(labels[x[0]],labels[y[0]])
            
        if str(labels[x[0]])==str(labels[y[0]]):
            return str(labels[x[0]])
      
    return str(labels[x[0]])+" "+str(labels[y[0]])
def predict2(model,text,cfggb):



    def ld_fun(data_list):
        tokenizer = lambda x: [y for y in x]
        pad_size = 32
        vocab_path= 'mydata/data/vocab3.pkl'
        vocab = pkl.load(open(vocab_path, 'rb'))
        contents = []
        
        for line in data_list:
            #lin = line.strip()
            try:
                content,label = line.split('\t')#content 为前面的新闻，label为后面标签
            except:
                print(line)
                print("dfshiou")
                input()
            words_line = []
            token = tokenizer(content)#tokenizer = lambda x: [y for y in x]，拆成单字
            seq_len = len(token)
            if pad_size:#短补长截,短的字符长度不变，长的截断为pad_size
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len))
        return contents    
    text = text.replace('\t','')
    
    content = text
    content = content+'\t'
    content = content+'0'
    sentences = []
    sentences.append(content)

    a = 'mydata'
    b = '123'
    config2 = Config(a,b)
    config2.n_vocab = cfggb

    mpd = ld_fun(sentences)
    mpd_iter = build_iterator(mpd, config2)
    
    # input()
    #labels = {0: '负向', 1: '正向'}
    labels = { 0:'like',1:'disgust',2:'happiness',3:'sadness',4:'anger',5:'surprise',6:'fear'}
    # labels = { 0:'正向',1:'负向',2:'正向',3:'负向',4:'负向',5:'负向',6:'负向'}
    model.eval()
    with torch.no_grad():
        for a,b in mpd_iter:
            outputs = model(a)
            x = torch.max(outputs.data, 1)[1].cpu().numpy()
        #     # y = torch.max(outputs.data, 1)[0].cpu().numpy()
        #     outputs.data[0][x[0]] = -999
        #     y = torch.max(outputs.data, 1)[1].cpu().numpy()
        #     print(labels[x[0]],labels[y[0]])
            
        # if str(labels[x[0]])==str(labels[y[0]]):
        #     return str(labels[x[0]])
      
    return str(labels[x[0]])    
if __name__ == '__main__':
    # GUI
    def fun(model):#按钮点击的函数
        f = open('result.csv', mode='w+', encoding='utf-8', newline='')
        fieldnames = ['评论','地区', '日期', '点赞','情绪']
        # csv_reader = csv.DictReader(f, fieldnames=fieldnames)
        writer = csv.writer(f)   
        writer.writerow(fieldnames)
        text3.insert(END,"模型加载...") 
        text = comment_input.get()
        # msg1 = sp.main(text)
        msg1 = "数据收集成功!"
        # input() 
        text2 = reprocfun(text)
        x = predict(model,text2)

        print(x)
        cont,prov,date,like = da.ana()
        emo = []
        for i in cont:
            x = predict(model,i)
            emo.append(x)
        for i in zip(cont,prov,date,like,emo):
            writer.writerow([i[0],i[1],i[2],i[3],i[4]])
        # print(like)
        # predict(model,x)
        text3.delete('1.0','1.20')
        msg1 = msg1+"\n"
        text3.insert(END,msg1) 
        time.sleep(2)
        text3.insert(END,msg1) 
        # ht.bro()       
        return 0


    a = 'mydata'
    b = '123'
    start_time = time.time()
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    config = Config(a,b)
    vocab_path= 'mydata/data/vocab3.pkl'
    vocab = pkl.load(open(vocab_path, 'rb'))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    config.n_vocab = len(vocab)
    global cfggb
    cfggb = len(vocab)
    root = Tk()
    root.title('demo')
    root.geometry('600x200+398+279')
    Label(root,text='微博链接',font=(15),fg='black').grid()
    Label2 = Label(root,text='情感倾向:',font=(15),fg='black')
    Label2.place(rely=0.5, relheight=0.1)
    comment_input=Entry(root,font=("微软雅黑",10),width=50)
    comment_input.place(rely=0.2,relheight=0.2)
    gx = comment_input.get() 
    text3 = Text(root)
    text3.place(rely=0.6, relheight=0.4)
    model = Model(config).to('cpu')
    print(model.parameters)
    model.load_state_dict(torch.load(config.save_path))#加载模型,定式
    B = Button(root,text ="OK", command = lambda:fun(model))
    B.place(relx=0.7, rely=0.2, relwidth=0.3, relheight=0.2)
    mainloop()

    




