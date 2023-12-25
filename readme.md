# 基于textcnn的舆情分析系统

## 目录格式

主目录/datasets/		训练模型所使用的数据集存放位置,模型基于textCNN魔改

主目录/graph/	和/pics/			没啥用,最开始用的matplot库,生成的图太丑了,废弃了

主目录/mydata/		存放模型训练时的词表和一些杂七杂八的文件

主目录下的各种文件	程序运行的所有代码以及测试样例等



## 主要代码文件介绍

htmltest.py	程序入口,启动程序就运行这个文件

webtest.py	 程序网页总框架,所有网页上的东西全在这

demo2.py	   情感分析模型,模型训练,推导全在这

spider.py		爬虫,用于爬取数据

dataanaly.py  数据处理

matplot.py	 可视化图表生成





## 使用

### 0.配置环境

使用requirements.yml把所需包安装好,python版本3.7.13

### 1.拿到cookie

在[微博移动版 (m.weibo.cn)](https://m.weibo.cn/)登录自己的账号拿到自己的cookie,并填在spider.py的cookie中

### 2.运行

运行htmltest.py



## 说明
