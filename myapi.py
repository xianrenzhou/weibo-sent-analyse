
"""
EasyDL 文本分类单标签 调用模型公有云API Python3实现
"""

import json
# import base64
import requests
"""
使用 requests 库发送请求
使用 pip（或者 pip3）检查我的 python3 环境是否安装了该库，执行命令
  pip freeze | grep requests
若返回值为空，则安装该库
  pip install requests
"""


# 目标文本的 本地文件路径，UTF-8编码，最大长度4096汉字
TEXT_FILEPATH = "test.txt"

# 可选的请求参数
# top_num: 返回的分类数量，不声明的话默认为 6 个
PARAMS = {"top_num": 2}

# 服务详情 中的 接口地址
MODEL_API_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/text_cls/zzznb"

# 调用 API 需要 ACCESS_TOKEN。若已有 ACCESS_TOKEN 则于下方填入该字符串
# 否则，留空 ACCESS_TOKEN，于下方填入 该模型部署的 API_KEY 以及 SECRET_KEY，会自动申请并显示新 ACCESS_TOKEN
ACCESS_TOKEN = ""
API_KEY = "usko8lAwl9szQwWQCeeNKtcO"
SECRET_KEY = "izOMzv0Ygtqu5TEb59bB5YnfAcLGVZI1"


print("1. 读取目标文本 '{}'".format(TEXT_FILEPATH))
with open(TEXT_FILEPATH, 'r',encoding='UTF-8') as f:
    text_str = f.read()
print("将读取的文本填入 PARAMS 的 'text' 字段")
PARAMS["text"] = text_str


if not ACCESS_TOKEN:
    print("2. ACCESS_TOKEN 为空，调用鉴权接口获取TOKEN")
    auth_url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"\
               "&client_id={}&client_secret={}".format(API_KEY, SECRET_KEY)
    auth_resp = requests.get(auth_url)
    auth_resp_json = auth_resp.json()
    ACCESS_TOKEN = auth_resp_json["access_token"]
    print("新 ACCESS_TOKEN: {}".format(ACCESS_TOKEN))
else:
    print("2. 使用已有 ACCESS_TOKEN")

'''
like
disgust
happiness
sadness
anger
surprise
fear
'''
print("3. 向模型接口 'MODEL_API_URL' 发送请求")
request_url = "{}?access_token={}".format(MODEL_API_URL, ACCESS_TOKEN)
response = requests.post(url=request_url, json=PARAMS)
response_json = response.json()
response_str = json.dumps(response_json, indent=4, ensure_ascii=False)
print("结果:\n{}".format(response_str))

print("like\ndisgust\nhappiness\nsadness\nanger\nsurprise\nfear")