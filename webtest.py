import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
# from streamlit_lottie import st_lottie
import json
from streamlit.elements.image import image_to_url
# image = Image.open('pics/pepie.jpg')
# resized_im = image.resize((round(image.size[0]*0.7), round(image.size[1]*0.7)))
st.set_page_config(layout="wide")
# img_url = image_to_url('bg.png',width=-3,clamp=False,channels='RGB',output_format='auto',image_id='',allow_emoji=False)

# st.markdown('''
# <style>
# .css-fg4pbf {background-image: url(''' + img_url + ''');}</style>
# ''', unsafe_allow_html=True)

# st.title("当前微博下的评论")
# df = pd.read_csv("result.csv")
# gb = GridOptionsBuilder.from_dataframe(df)
# gb.configure_pagination(paginationAutoPageSize=False,paginationPageSize=10)
# gridOptions = gb.build()
# AgGrid(df, gridOptions=gridOptions)
# st.image(resized_im, caption='Sunrise by the mountains', use_column_width=False)
import streamlit as st
import pandas as pd
import numpy as np
import demo2 as dm2
import csv
import matplot as mt
import streamlit.components.v1 as components
import pyecharts
f11 = open("mid.txt",'r',encoding="utf-8")
i = f11.readline()
print("主程序加载")
def main():
    # def tmain(text):
    a = 'mydata'
    b = '123'
    # flname = "result2.csv"
    print("主函数开始")
    start_time = dm2.time.time()
    dm2.torch.manual_seed(1)
    dm2.torch.cuda.manual_seed_all(1)
    dm2.torch.backends.cudnn.deterministic = True 
    config = dm2.Config(a,b)
    vocab_path= 'mydata/data/vocab3.pkl'
    vocab = dm2.pkl.load(open(vocab_path, 'rb'))
    time_dif = dm2.get_time_dif(start_time)
    print("Time usage:", time_dif)
    config.n_vocab = len(vocab)
    global cfggb
    cfggb = len(vocab)
    model = dm2.Model(config).to('cpu')
    # print(model.parameters)
    model.load_state_dict(dm2.torch.load(config.save_path))#加载模型,定式
    #     return dm2.predict(model,text,cfggb)
    # def test2(text):
    #     global res1
    #     res1 = tmain(text)
    #     print("test2")
    # def test3(a):
    #     a = 1
    #     print("test3")
    side = ["主界面","数据分析结果"]
    st.sidebar.title("使用左侧面板")
    la = st.sidebar.selectbox("请选择",side)
    # la = "页面2"
    a = 0
    if la == "主界面":
        r1,r2,r3 = st.columns([3,6,3])
        with r2:
            st.title("情绪分析(单句)")
            
            test = st.text_input("请输入文字:", key="请输入内容")
            x = st.button("开始分析")
            res = ""
            if x:
                print("x")
                res = dm2.predict(model,test,cfggb)
            # res = tmain(test)
            # st.write("情感倾向为:"+res)
            st.write("情感倾向为:   "+res)
        
            st.title("情绪分析(本地数据集/链接)")
            text = st.text_input("请输入微博链接或文件地址:如(https://m.weibo.cn/detail/4835751314857743)", key="请输入内容")
            sel = ["啥也不选","已有数据集","在线爬取数据集"]
            spy = st.radio("这是单选按钮", sel)
            y = st.button("开始执行")
            if y:
                if spy == "啥也不选":
                    st.write("你啥也没选")
                elif spy == "已有数据集":
                    st.write("模型加载完毕,开始推导数据")
                    # msg1 = dm2.sp.main(text)
                    text2 = dm2.reprocfun(text)
                    x = dm2.predict2(model,text2,cfggb)
                    st.write("数据推导完毕,开始分析数据")
                    print(x)
                    cont,prov,date,like = dm2.da.ana()
                    st.write("数据分析完毕,正在进行可视化")
                    emo = []
                    for i in cont:
                        x = dm2.predict2(model,i,cfggb)
                        emo.append(x)
                    # for i in zip(cont,prov,date,like,emo):
                    #     writer.writerow([i[0],i[1],i[2],i[3],i[4]])
                    if text == "":
                        text = "result2.csv"
                    mt.fun(text)
                    st.write("可视化图表生成完毕,请在左边选择数据分析结果查看")
                    st.title("评论内容")
                    df1 = pd.read_csv(text)
                    st.dataframe(df1,width=1500)
                    f11 = open("mid.txt",'w+',encoding="utf-8")
                    f11.write(text)
                else:
                    st.write("链接获取成功,开始爬取数据")
                    f = open('result.csv', mode='w+', encoding='utf-8', newline='')
                    fieldnames = ['评论','地区', '日期', '点赞','情绪']
                    writer = csv.writer(f)   
                    writer.writerow(fieldnames)
                    
                    msg1 = dm2.sp.main(text)
                    st.write("数据获取成功")
                    st.write("模型加载完成,开始推导")
                    # text2 = dm2.reprocfun(text)
                    # x = dm2.predict2(model,text2,cfggb)
                    st.write("数据推导完毕,开始分析数据")
                    st.write(msg1)
                    print(x)
                    cont,prov,date,like = dm2.da.ana()
                    st.write("数据分析完毕,正在进行可视化")
                    emo = []
                    for i in cont:
                        x = dm2.predict2(model,i,cfggb)
                        emo.append(x)
                    for i in zip(cont,prov,date,like,emo):
                        writer.writerow([i[0],i[1],i[2],i[3],i[4]])
                    mt.fun()
                    cont = []
                    prov = []
                    date = []
                    like = []
                    emo = []
                    st.write("可视化图表生成完毕,请在左边选择数据分析结果查看")               
                    st.title("评论内容")
                    df1 = pd.read_csv("result.csv")
                    st.dataframe(df1,width=1500)
                    # global flname 
                    f22 = open("mid.txt",'w+',encoding="utf-8")
                    print("改成了result")
                    f22.write("result.csv")
                    # print(like)
                    # predict(model,x)
                    # text3.delete('1.0','1.20')
                    # msg1 = msg1+"\n"
                    # text3.insert(END,msg1) s
                    # time.sleep(2)
                    # text3.insert(END,msg1) 
                    # ht.bro()       
        
    
            
            if a == 1:
                df = pd.read_csv("result.csv")
                st.dataframe(df)  # st.dataframe(df)可以用st.write(df)来代替，效果一样
    elif la == "数据分析结果":
        env=["分析总览","数据总览"]
        la2 = st.sidebar.selectbox("请选择模块",env)
        # if la2 == "词云":
        #     st.title("词云")
        #     image = Image.open('pics/wdc.png')
        #     st.image(image, caption='wordcloud', use_column_width=False)
        if la2 == "分析总览":
            st.title("分析总览")
            st.title(" ")
            st.title(" ")
            r0,r1= st.columns([3,3])
            with r1:
                st.write("情绪-点赞关系图")
                with open("emolk.html") as fp1:
                    text001=fp1.read()
                components.html(text001,height=400,width=700)
            with r0:
                st.write("各情绪人数统计图")
                with open("procnts.html") as fp2:
                    text001=fp2.read()
                components.html(text001,height=400,width=700)
            # r2,r3,r4 = st.columns([2,3,2])
            # with r3:
            st.write("     ")
            st.write("     ")
            st.write("     ")
            st.write("情绪-地区折线图")
            with open("nwpe.html") as fp3:
                text001=fp3.read()
            components.html(text001,height=500,width=1400)
            st.write("     ")
            st.write("     ")
            st.write("     ")
            r3,r4 ,r5= st.columns([3,3,2])
            with r3:
                st.write("情绪分布雷达图")
                with open("radar.html") as fp2:
                    text001=fp2.read()
                components.html(text001,height=500,width=900)                
            with r4:
                st.write("情绪分布饼状图")
                with open("pie_rich_label.html") as fp2:
                    text001=fp2.read()
                components.html(text001,height=500,width=900)
            st.write("     ")
            st.write("     ")
            st.write("     ")
            r6,r7 = st.columns([1,1])
            with r6:
                st.write("热力地图")
                with open("new_map.html") as fp2:
                    text001=fp2.read()
                components.html(text001,height=600,width=900)
            with r7:
                st.write("词云")
                image = Image.open('pics/wdc.png')
                st.image(image,  use_column_width=False)
            # with r0:
            #     st.title("情绪分布")
      
            # with open("pie_rich_label.html") as fp: #如果遇到decode错误，就加上合适的encoding
            #     text333=fp.read()
            # print(text333)
            # components.html(text333,height=500,width=1000)
            # with r12:
            #     st.write("|\n|\n|")
            # text2=""
            # with open("emolk.html") as fp2:
            #     text444=fp2.read()
            # components.html(text444,height=500,width=1000)
            # components.html(text444,height=500,width=1000)
            # components.html(text444,height=500,width=1000)
            # components.html(text444,height=500,width=1000)
            # components.html(text444,height=500,width=1000)
            # components.html(text444,height=500,width=1000)

        # if la2 == "情绪地区关系折线图":
        #     st.title("情绪-地区")
        #     image = Image.open('pics/pe_2.jpg')
        #     st.image(image, caption='area&emotion',use_column_width=True)
        # if la2 == "情绪点赞关系柱状图":
        #     st.title("情绪-点赞")
        #     image = Image.open('pics/lk_emo.jpg')
        #     resized_im = image.resize((round(image.size[0]*0.7), round(image.size[1]*0.7)))
        #     st.image(resized_im, caption='like&emotion', use_column_width=False)        
        if la2 == "数据总览":
            r9,r10 = st.columns([1,3])
            with r10:
                st.title("评论内容")
                f11 = open("mid.txt",'r',encoding="utf-8")
                flname = f11.readline()
                print(flname)
                df1 = pd.read_csv(flname)
                st.dataframe(df1,width=1500,height=1500)
        

   
main()