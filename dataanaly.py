# # mondic={Jan:}
# f = open("spider.csv",'r',encoding='UTF-8')
# x = f.readlines()
# for i in x:
#     print(i)

def ana():
    contents = []
    province = []
    date = []
    like = []
    f = open("spider.csv",'r',encoding='UTF-8')
    x = f.readlines()
    for i in x:
        # z = 0
        # 
        z = i.split(',')
        if z[0] =="":
            continue
        else:
            contents.append(z[0])
            province.append(z[1])
            date.append(z[2])
            like.append(z[3].replace("\n",""))
            # if(z == 0):
            #     contents.append(j)
            # elif(z == 1):
            #     province.append(j)
            # elif(z == 2):
            #     date.append(j)
            # elif(z == 3):
            #     j = int(j)
            #     like.append(j)
            
            # z = z+1
    # print(province)
    # contents.append("宝贝害怕了吗")
    # province.append("安徽")
    # date.append("0")
    # like.append(0)
    return contents,province,date,like
