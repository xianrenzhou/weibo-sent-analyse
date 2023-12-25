import matplotlib.pyplot as plt
import csv
import numpy as np
import jieba
import wordcloud
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.charts import Bar
def isinDict(dict,tstr):
    if (tstr in dict) == True:
        dict[tstr] += 1
    else:
        dict[tstr] = 1

def emosplit(str,dict):
    print(str)
    j = str.split(',')[4]
    k = j.split('\n')[0]
    isinDict(dict,k)

def province(str,dict):
    j = str.split(',')[4]
    emo = j.split('\n')[0]

    province = str.split(',')[1]
    isinDict(dict[emo],province)
    # if emo == 'like':

def emograpic(dic):
    a=[]
    b=[]
    for key in dic:
        a.append(key)        
        b.append(dic[key])
    size=[]
    t=sum(b)#统计总的发表篇幅
    label=a
    # plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
    #计算每种类型所占的比例
    for u in b:
        i=u/t
        size.append(i)
        # plt.plot(size)
    # plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman 
    # plt.title("情绪分布饼状图", fontsize=15,fontweight='bold')
    colors = ["#5D7599","#ABB6C8","#DADADA","#F7F0C6","#C2C4B6","#B6B4C2","#AAC9CE"]
    # plt.figure(figsize = (10,10))  
    # explode = (0.02,0.03,0.04,0.05,0.06,0.07,0.08)
    
    # plt.pie(size, explode=explode,colors=colors, labels=label,
    #        autopct = '%1.2f%%',pctdistance = 1.1,labeldistance = 1.2)
  
    # plt.legend(loc='upper right')
    # plt.title("情绪分布饼状图", fontsize=25,fontweight='bold') 
    # plt.savefig("pics/pepie.jpg")
    # plt.show()
    testttt = [list(z) for z in zip(a,b)]
    from pyecharts.charts import Radar
    j = 0
    for i in b:
        if i > j:
            j = i
    print(b,j)

    z = a
    print(z)
    radar = Radar(init_opts=opts.InitOpts(
                                width='700px',
                                height='500px',
                                page_title='page',
                                ))
    radar.add_schema(
             schema=[
            opts.RadarIndicatorItem(name=z[0], max_=j),
            opts.RadarIndicatorItem(name=z[1], max_=j),
            opts.RadarIndicatorItem(name=z[2], max_=j),
            opts.RadarIndicatorItem(name=z[3], max_=j),
            opts.RadarIndicatorItem(name=z[4], max_=j),
            opts.RadarIndicatorItem(name=z[5], max_=j),
            opts.RadarIndicatorItem(name=z[6], max_=j),
        ],
        splitarea_opt=opts.SplitAreaOpts(
        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
        ),
        textstyle_opts=opts.TextStyleOpts(color="#000"),
    ).set_series_opts(label_opts=opts.LabelOpts(is_show=True,color=["#4e79a7"])).add("人数雷达图", [b],color=["#4e79a7"]).set_global_opts(
       legend_opts=opts.LegendOpts()
    ).render("radar.html")
    css = (
    Pie( init_opts=opts.InitOpts(
                                width='900px',
                                height='500px',
                                page_title='page',
                                ))
    .add(
        "",
        testttt,
        radius=["40%", "55%"],
        label_opts=opts.LabelOpts(
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

    .render("pie_rich_label.html")
    
)
    
    # from pyecharts.charts import Grid
   
    # # grid = Grid()
    # # grid.add(c, grid_opts=opts.GridOpts(pos_left = 20))
    # # grid.add(radar,  grid_opts=opts.GridOpts(pos_right = 20))
    # # grid.render("text.html")
    # grid = (
    # Grid()
    # .add(css, grid_opts=opts.GridOpts(pos_left="55%"))
    # .add(radar, grid_opts=opts.GridOpts(pos_right="55%"))
    # .render("grid_horizontal.html")

def like_emo(str,lk_em_dict):
    # lk_em_dict = {}s
    j = str.split(',')[4]
    emo = j.split('\n')[0]
    lkcnt = str.split(',')[3]
    if (emo in lk_em_dict) == True:
        lk_em_dict[emo] += int(lkcnt)
    else:
        lk_em_dict[emo] = int(lkcnt)
    


def lk_em_graphic(dic):
    a=[]
    b=[]
    
    for key in dic:
        a.append(key)        
        b.append(dic[key])
    size=[]
    t=sum(b)#统计总的发表篇幅
    label=a

    plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
    #计算每种类型所占的比例
    for u in b:
        size.append(u)
        # plt.plot(size)
    colors = ["#5D7599","#ABB6C8","#DADADA","#F7F0C6","#C2C4B6","#B6B4C2","#AAC9CE"]
    plt.figure(figsize = (10,10))  
    plt.title("情绪-点赞关系柱状图", fontsize=15,fontweight='bold')
    plt.xlabel("情绪", fontsize=15,fontweight='bold')
    plt.ylabel("总点赞数", fontsize=15,fontweight='bold')
    # 修改坐标轴字体及大小
    plt.yticks(fontproperties='Times New Roman', size=15,weight='bold')#设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=15)


    # 设置标题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.tight_layout()  # 解决绘图时上下标题重叠现象

    #画线
    # plt.vlines(starts2time, min(Mfcc1)-10, max(Mfcc1)+10, colors="black", linestyles="solid",lw=2)
    # plt.vlines(ends2time, min(Mfcc1)-10, max(Mfcc1)+10, colors="black", linestyles="dashed",lw=2.5)
    plt.bar(a, b, color=colors, width=0.5)
    # plt.plot(a,b)
    plt.savefig("pics/lk_emo.jpg",bbox_inches='tight',pad_inches=0.0) 
    # plt.show()
    c = (
        Bar(init_opts=opts.InitOpts(
                                width='700px',
                                height='400px',
                                page_title='page',
                                )).add_xaxis(a)
        .add_yaxis("点赞数",b)
        .render("emolk.html")

    )

def pe_graphic(dic):
    pe_emo = {}
    allprovince = {}
    cnt = 0
    for i in dic:
        for j in dic[i]:
            print(j)
            if (j in allprovince) == True:
                allprovince[j]+=dic[i][j]
                # cnt+=dic[i][j]
            else:
                allprovince[j] = dic[i][j]
                # cnt+=dic[i][j]
    for i in allprovince:
        pe_emo[i] = {}
        for j in dic:
            pe_emo[i][j] = 0
    for i in dic:
        for j in dic[i]:
            pe_emo[j][i]+=dic[i][j]
    x_gra = []
    y_gra = {}
    y_gra2 = {}
    a = []
    b = []
    print(allprovince)
    for ia in allprovince:
        a.append(ia)
        b.append(allprovince[ia])
    c = (
        Bar(init_opts=opts.InitOpts(
                                width='700px',
                                height='400px',
                                page_title='page',
                                )).add_xaxis(a)
        .add_yaxis("各省人数",b)
        .render("procnts.html")

    )
    # from pyecharts import Radar






    for i in pe_emo:
        # print[i]
        x_gra.append(i)
        for j in pe_emo[i]:
            print(j)
            # y_gra[j].append(i[j])
            if (j in y_gra) == True:
                y_gra[j].append(pe_emo[i][j]/allprovince[i])
                y_gra2[j].append(pe_emo[i][j])
            else:
                y_gra[j] = [pe_emo[i][j]/allprovince[i]]
                y_gra2[j] = [pe_emo[i][j]]
            # print(allprovince)
    # print(allprovince,cnt,pe_emo)
    plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
    plt.figure(figsize = (13,7))  
    plt.ylim((0, 100))
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman
    plt.rcParams['axes.unicode_minus'] = False 
    plt.title("情绪-省份关系折线图", fontsize=15,fontweight='bold')
    plt.xlabel("省份", fontsize=15,fontweight='bold')
    plt.ylabel("情感倾向占比", fontsize=15,fontweight='bold')
    # 修改坐标轴字体及大小
    
    from pyecharts.charts import Line
    line=(
    Line(init_opts=opts.InitOpts(
                                width='1400px',
                                height='500px',
                                page_title='page',
                                ))
    .set_global_opts(
        # tooltip_opts=opts.TooltipOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=opts.DataZoomOpts(is_show=True, type_ = "slider",)
    )
    .add_xaxis(xaxis_data=x_gra)
    .add_yaxis(
        series_name="like",
        y_axis=y_gra["like"], is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="disgust",
        y_axis=y_gra["disgust"], is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="surprise",
        y_axis=y_gra["surprise"], is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="anger",
        y_axis=y_gra["anger"], is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="fear",
        y_axis=y_gra["fear"], is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="sadness",
        y_axis=y_gra["sadness"], is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="happiness",
        y_axis=y_gra["happiness"], is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    ) 
    .render("nwpe.html")
)
    from pyecharts.charts import Map,Timeline
    c1 = (
        Map()
        .add("like", [list(z) for z in zip(x_gra, y_gra2["like"])], "china")
        .set_global_opts(
           
           
        visualmap_opts=opts.VisualMapOpts(max_=max(y_gra2["like"]),
                                      min_=min(y_gra2["like"]),
                                      range_color=["#EFEFEF","#BEFB66","#EB672F","#EA3725"]
                                     ),
        )
    )
    c2 = (
        Map()
        .add("disgust", [list(z) for z in zip(x_gra, y_gra2["disgust"])], "china")
        .set_global_opts(
           
           
        visualmap_opts=opts.VisualMapOpts(max_=max(y_gra2["disgust"]),
                                      min_=min(y_gra2["disgust"]),
                                      range_color=["#DEDEF7","#BEFB66","#EB672F","#EA3725"]
                                     ),
        )
    )
    c3 = (
        Map()
        .add("surprise", [list(z) for z in zip(x_gra, y_gra2["surprise"])], "china")
        .set_global_opts(
           
           
        visualmap_opts=opts.VisualMapOpts(max_=max(y_gra2["surprise"]),
                                      min_=min(y_gra2["surprise"]),
                                      range_color=["#DEDEF7","#BEFB66","#EB672F","#EA3725"]
                                     ),
        )
    )
    c4 = (
        Map()
        .add("anger", [list(z) for z in zip(x_gra, y_gra2["anger"])], "china")
        .set_global_opts(
           
           
        visualmap_opts=opts.VisualMapOpts(max_=max(y_gra2["anger"]),
                                      min_=min(y_gra2["anger"]),
                                      range_color=["#DEDEF7","#BEFB66","#EB672F","#EA3725"]
                                     ),
        )
    )
    c5 = (
        Map()
        .add("fear", [list(z) for z in zip(x_gra, y_gra2["fear"])], "china")
        .set_global_opts(
           
           
        visualmap_opts=opts.VisualMapOpts(max_=max(y_gra2["fear"]),
                                      min_=min(y_gra2["fear"]),
                                      range_color=["#DEDEF7","#BEFB66","#EB672F","#EA3725"]
                                     ),
        )
    )
    c6 = (
        Map()
        .add("sadness", [list(z) for z in zip(x_gra, y_gra2["sadness"])], "china")
        .set_global_opts(
           
           
        visualmap_opts=opts.VisualMapOpts(max_=max(y_gra2["sadness"]),
                                      min_=min(y_gra2["sadness"]),
                                      range_color=["#DEDEF7","#BEFB66","#EB672F","#EA3725"]
                                     ),
        )
    )
    c7 = (
        Map()
        .add("happiness", [list(z) for z in zip(x_gra, y_gra2["happiness"])], "china")
        .set_global_opts(
           
        visualmap_opts=opts.VisualMapOpts(max_=max(y_gra2["happiness"]),
                                      min_=min(y_gra2["happiness"]),
                                      range_color=["#DEDEF7","#BEFB66","#EB672F","#EA3725"]
                                     ),
        )
    )
    timeline = Timeline()
    timeline.add(c1,"like")
    timeline.add(c2,"disgust")
    timeline.add(c3,"surprise")
    timeline.add(c4,"anger")
    timeline.add(c5,"fear")
    timeline.add(c6,"sadness")
    timeline.add(c7,"happiness")
    timeline.render("new_map.html")
    plt.plot(x_gra, y_gra["like"], marker='o', markersize=7,label = "like") 
    plt.plot(x_gra, y_gra["disgust"], marker='o', markersize=7,label = "disgust",linewidth = "7",color = 'red')
    plt.plot(x_gra, y_gra["surprise"], marker='o', markersize=7,label ="surprise" ) 
    plt.plot(x_gra, y_gra["anger"], marker='o', markersize=7,label = "anger")   
    plt.plot(x_gra, y_gra["fear"], marker='o', markersize=7,label = "fear")
    plt.plot(x_gra, y_gra["sadness"], marker='o', markersize=7,label = "sadness") 
    plt.plot(x_gra, y_gra["happiness"], marker='o', markersize=7,label = "happiness",linewidth = "7",color = "pink")   
    plt.tight_layout() 
    plt.legend()

    plt.savefig("pics/pe_2.jpg")
    plt.show()
    print(x_gra,y_gra)

def wdc(str):
    ls = jieba.lcut(str) # 生成分词列表
    text = ' '.join(ls) # 连接成字符串


    stopwords = ["我","你","她","的","是","了","在","也","和","就","都","这","啊啊啊"] # 去掉不需要显示的词
    import numpy as np
    from PIL import Image
    mk = np.array(Image.open("wdbg.png"))
    wc = wordcloud.WordCloud(font_path="msyh.ttc",
               mask = mk,
               width = 500,
               height = 500,
               background_color='white',
               max_words=200,
               stopwords=stopwords)
    # msyh.ttc电脑本地字体，写可以写成绝对路径
    wc.generate(text) # 加载词云文本
    wc.to_file("pics/wdc.png") # 保存词云文件

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
    flag = 0
    for i in f.readlines():
        if flag == 1:
            emosplit(i,counts)
            province(i,pe)
            like_emo(i,lk_em)
            str += i.split(',')[0]
        flag = 1
    # print(counts)
    # print(lk_em)
    # print(pe)
    # print(str)
    pe_graphic(pe)
    emograpic(counts)
    lk_em_graphic(lk_em)
    wdc(str)