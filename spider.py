
import csv
import requests
import time
 
headers = {
    'Cookie': "",
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
 
}
 
url = 'https://m.weibo.cn/comments/hotflow'
f = open('spider.csv', mode='w+', encoding='utf-8', newline='')
writer = csv.writer(f)
 
 
def get(msgid):
    data = {
        'id': msgid,
        'mid': msgid,
        'max_id_type': 0,
    }
    print(type(data['id']))
    # input()
    resp = requests.get(url=url, headers=headers, params=data).json()
    max_id = resp['data']['max_id']
    mid_t = resp['data']['max_id_type']
    data_list = resp['data']['data']
    for dicts in data_list:
        
        testt = dicts['source'].replace("来自","")
        # print(testt)
        # input()

        # user_name = dicts['user']['screen_name']  # 用户名
        like_count = dicts['like_count']  # 点赞该评论数
        text = dicts['text'].split('<')[0].replace(",","z")  # 评论
        # user_url = dicts['user']['profile_url']  # 用户微博链接
        created_at = dicts['created_at']  # 评论时间
        writer.writerow([text,testt, created_at,like_count])
        # print(1)
        # time.sleep(3)  # 睡一下
        # print(2)
        # print( text + " "+ str(like_count)+" "+ testt +" "+str(created_at))
    # input()
    get2(max_id,msgid,mid_t)
 
 
def get2(max_id,msgid,mid_t):
    a = 1
    mid_t = mid_t
    while True:
        data2 = {
            'id': msgid,
            'mid': msgid,
            'max_id': max_id,
            'max_id_type': mid_t
        }
        # resp2 = requests.get(url=url, headers=headers, params=data2).json()
        resp2 = requests.get(url=url, headers=headers, params=data2).json()
        print(resp2)
        # input()
        max_id = resp2['data']['max_id']
        mid_t = resp2['data']['max_id_type']
        # print(resp2)
        data_list = resp2['data']['data']
        for dicts in data_list:
            testt = dicts['source'].replace("来自","")
            # user_name = dicts['user']['screen_name']  # 用户名
            like_count = dicts['like_count']  # 点赞该评论数
            text = dicts['text'].split('<')[0].replace(",","")  # 评论
            # user_url = dicts['user']['profile_url']  # 用户微博链接
            created_at = dicts['created_at']  # 评论时间
            writer.writerow([text,testt, created_at,like_count])
            # print( text + " "+ str(like_count)+" "+ testt +" "+str(created_at))
        if a == 100:  # 我没爬完，10页左右
            break
        if data2['max_id_type'] != 0:
            break
        a += 1
 
 
 
def main(spidermsg):
    headers['Referer'] = spidermsg
    msgid = headers['Referer'].split('detail/')[1]
    get(int(msgid))
    return "数据收集成功!"
 
 
# if __name__ == '__main__':
#     main()


'''

'''