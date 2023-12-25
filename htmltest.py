import os
import sys

# os.system(""C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" http://www.baidu.com")
def bro():
    abspath = sys.path[0]
    mycommamd = "streamlit run "+abspath.replace("\\",'/') +"/webtest.py"
    print(mycommamd)
    os.system(mycommamd)
    # os.system("start http://localhost:8501")

# webbrowser.open_new_tab('file://helloworld.html')]
bro()