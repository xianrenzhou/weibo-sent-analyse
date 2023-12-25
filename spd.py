import csv
import requests
import time
 
headers = {
    #'Cookie': 'SCF=AiRBUT9QJuus8YUaFkCqXwvCsmhD34I2T2iaqq8ckshEWZYqziVi9kdSr1Pq9VTP_EAKxArt8_tW-JCQRPVKyJ8.; SUB=_2A25M4vtaDeRhGeFJ6FQR9SrIzDWIHXVsLIUSrDV6PUJbktCOLVTckW1NfA3sqYESNrYjdiWqi7X77v0RmHWhXyN6; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WW88ZW9w-vn-iGnnPe7fqwT5JpX5K-hUgL.FoMNe0q7SKBXS0.2dJLoI7vbIPHbIPHbIPHkeK-4S0zt; _T_WM=14995490976; WEIBOCN_FROM=1110006030; MLOGIN=1; M_WEIBOCN_PARAMS=oid%3D4735010203238477%26luicode%3D20000061%26lfid%3D4735010203238477%26uicode%3D20000061%26fid%3D4735010203238477; XSRF-TOKEN=0b6a51',
    'Cookie':'SCF=AokFWp8zCf3UA8mykfnNqhQtuzr6IMxLAI4EfQL3Pi6Op4PQjjDSYONISoPTmNZFMNzVmSBhGEF-LXNeZd0B7uU.; ALF=1670838661; __bid_n=1846b4285fb3cb2c534207; FPTOKEN=30$zDUnJmhgj8NGNtw+Tdlv9nDYabQyALGHXgJUinVhsSDFP/5/X8Ro5kkj/csEkOiPDmcViliVwy1pP6bDHRXNhCgZ8kJJRj1qcGEkYoeCluakMlkgWa4issCh62AAI/n2+XwVPB22YCF6RqH1f67ZTCUvOrG0ELzK3HLUgLrABGHZbMRD7e2btlSux74JQjOiBhIdm8qjRm8GZ6NTSDL45KbBOTba8TOZew0jxBcvDfXZ3fIBQ82hhq5Xj7r74oh6qqHpSLGMl+RNGuPn9yJsyPCW72JCGthF5h3Nsby0dr0/eyj6BZOYC433brQKfOeaF6ho45USKPi1yPe195RSRU8fRSZK9+/TCdh4RD7OMck9yKpMhYSPJaSYtaI9QL+E|f1XrRprtsaAVdk73LVLKviFdw4Rb6u81njwBR2X1Oic=|10|ec5534c8f9c9c6fca63df665f7cb5a18; SUB=_2A25OawlEDeRhGeFM6loQ9y3MzTiIHXVtl5cMrDV6PUJbkdANLW3GkW1NQPRnBwJNf-XWjuX23VuRN0L6eXo71C1c; _T_WM=23307338514; MLOGIN=1; XSRF-TOKEN=285636; WEIBOCN_FROM=1110006030; OUTFOX_SEARCH_USER_ID_NCOO=946027512.9722204; M_WEIBOCN_PARAMS=oid%3D4527007570168851%26luicode%3D20000061%26lfid%3D4527007570168851',
    'Referer': 'https://m.weibo.cn/detail/4735010203238477',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
 
}
 
url = 'https://m.weibo.cn/comments/hotflow'
f = open('微博正文评论多页获取2.csv', mode='a', encoding='utf-8', newline='')
writer = csv.writer(f)
 
 
def get():
    data = {
        'id': 4735010203238477,
        'mid': 4735010203238477,
        'max_id_type': 0,
    }
    resp = requests.get(url=url, headers=headers, params=data).json()
    # resp2 = requests.get(url=url, headers=headers, params=data2)
    print(resp)
    
    max_id = resp['data']['max_id']
    data_list = resp['data']['data']
    for dicts in data_list:
        user_name = dicts['user']['screen_name']  # 用户名
        like_count = dicts['like_count']  # 点赞该评论数
        text = dicts['text'].split('<')[0]  # 评论
        user_url = dicts['user']['profile_url']  # 用户微博链接
        created_at = dicts['created_at']  # 评论时间
        writer.writerow([user_name, like_count, text, user_url, created_at])
        # time.sleep(3)  # 睡一下
    input()
    get2(max_id)
 
 
def get2(max_id):
    a = 1
    while True:
        data2 = {
            'id': 4735010203238477,
            'mid': 4735010203238477,
            'max_id': max_id,
            'max_id_type': 0
        }
        # resp2 = requests.get(url=url, headers=headers, params=data2).json()
        resp2 = requests.get(url=url, headers=headers, params=data2).json()
        print(resp2)
        input()
        max_id = resp2['data']['max_id']
        # print(resp2)
        data_list = resp2['data']['data']
        for dicts in data_list:
            user_name = dicts['user']['screen_name']  # 用户名
            like_count = dicts['like_count']  # 点赞该评论数
            text = dicts['text'].split('<')[0]  # 评论
            user_url = dicts['user']['profile_url']  # 用户微博链接
            created_at = dicts['created_at']  # 评论时间
            writer.writerow([user_name, like_count, text, user_url, created_at])
 
        if a == 5:  # 我没爬完，10页左右
            break
        a += 1
 
 
def main():
    get()
 
 
if __name__ == '__main__':
    main()