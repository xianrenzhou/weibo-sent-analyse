
#pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyecharts


import matplotlib.pyplot as plt
import csv
import numpy as np
import jieba
import wordcloud
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.charts import Bar

def isinDict(dict,tstr):  #判断字符串是否在字典里,在的话键(字符串)对应的值加一,不在的话加到字典里,并且值设为一
    if (tstr in dict) == True:
        dict[tstr] += 1
    else:
        dict[tstr] = 1

def emosplit(str,dict):  #把str中的情绪词拆分出来并统计各个情绪的人数,存到dic中,dic是一个字典,形式为{like:2,fear:3 ...}
    print(str)
    j = str.split(',')[4]
    k = j.split('\n')[0]
    isinDict(dict,k)

def emograpic(dic):
    a=[]  #存放情绪
    b=[]  #存放情绪对应的人数
    for key in dic:
        a.append(key)        
        b.append(dic[key])


    size=[]   #存放比例
    t=sum(b)  #统计总人数
    label=a
    # plt.rcParams['font.sans-serif']=['Microsoft JhengHei']


    #计算每种类型所占的比例
    for u in b:
        i=u/t
        size.append(i)
        # plt.plot(size)
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman 
    plt.title("情绪分布饼状图", fontsize=15,fontweight='bold')
    colors = ["#5D7599","#ABB6C8","#DADADA","#F7F0C6","#C2C4B6","#B6B4C2","#AAC9CE"]  #饼图各个区域颜色
    plt.figure(figsize = (10,10))  #画布大小  
    explode = (0.02,0.03,0.04,0.05,0.06,0.07,0.08)  #用于控制饼图是否"炸开"
    
    plt.pie(size, explode=explode,colors=colors, labels=label,
           autopct = '%1.2f%%',pctdistance = 1.1,labeldistance = 1.2)
  
    plt.legend(loc='upper right')
    plt.title("情绪分布饼状图", fontsize=25,fontweight='bold') 
    plt.savefig("testpie.jpg")
    plt.show()



    #进阶方法  pyecharts 包
    testttt = [list(z) for z in zip(a,b)]  #情绪和人数用新的格式存放 
    #print(testttt)

    css = (
    Pie( init_opts=opts.InitOpts(  #控制大小
                                width='900px',
                                height='500px',
                                page_title='page',
                                ))
    .add(
        "",
        testttt,
        radius=["40%", "55%"],#饼状图变为圆环图
        label_opts=opts.LabelOpts(  #富文本
            position="outside",
            formatter="\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
            background_color="#eee",
            border_color="#aaa",
            border_width=1,
            border_radius=4,
            rich={
                "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                "abg": {
                    "backgroundColor": "#e3e3e3",
                    "width": "100%",
                    "align": "right",
                    "height": 22,
                    "borderRadius": [4, 4, 0, 0],
                },
                "hr": {
                    "borderColor": "#aaa",
                    "width": "100%",
                    "borderWidth": 0.5,
                    "height": 0,
                },
                "b": {"fontSize": 16, "lineHeight": 33},
                "per": {
                    "color": "#eee",
                    "backgroundColor": "#334455",
                    "padding": [2, 4],
                    "borderRadius": 2,
                },
            },
        ),
    )

    .render("test2.html")  #保存
    
)

def fun(flname = 'result.csv'):
    counts = {'happiness':0, 'like':0, 'anger':0, 'surprise':0, 'disgust':0, 'sadness':0,'fear':0}
    lk_em = {'happiness':0, 'like':0, 'anger':0, 'surprise':0, 'disgust':0, 'sadness':0,'fear':0}
    pe = { 'like':{
        
    },'disgust':{
       
    },'happiness':{
        
    },'sadness':{
        
    },'anger':{
       
    },'surprise':{
       
    },'fear':{
        
    }}
    str = ""
    if flname == '':
        flname = 'result2.csv'
    f = open(flname, mode='r', encoding='utf-8')
    flag = 0  #文件第一行是标题行,做个标记,忽略标题行
    for i in f.readlines():
        if flag == 1:
            emosplit(i,counts)
            # province(i,pe)
            # like_emo(i,lk_em)
            # str += i.split(',')[0]
        flag = 1
    # print(counts)
    # print(lk_em)
    # print(pe)
    # print(str)
    # pe_graphic(pe)
    emograpic(counts)
    # lk_em_graphic(lk_em)
    # wdc(str)
fun()