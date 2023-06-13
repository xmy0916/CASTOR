import pynvml
pynvml.nvmlInit()
import time
import os
#from send_email import send_msg

import smtplib
from email.mime.text import MIMEText
from email.header import Header
 
def send_msg(target_email,msg):
  sender = 'from@runoob.com'
  receivers = [target_email]  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
 
  # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
  message = MIMEText(msg, 'plain', 'utf-8')
  subject = 'nvidia显卡监控'
  message['Subject'] = Header(subject, 'utf-8')
 
 
  try:
      smtpObj = smtplib.SMTP('localhost')
      smtpObj.sendmail(sender, receivers, message.as_string())
      print("邮件发送成功")
  except smtplib.SMTPException:
      print("Error: 无法发送邮件")


def watch_nvidia(nvidia_ids,min_memory):
  flag = [1 for i in nvidia_ids]
  for i in nvidia_ids:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("card {} free mem is {}".format(i,meminfo.free * 1.0 /(1024**3)))
    if meminfo.free * 1.0 /(1024**3) > min_memory:
      flag[i-1]=0
    else:
      flag[i-1]=1
  if 1 not in flag:
    print("nvidia free")
    return 1
  else:
    print("nvidia busy")
    return 0


    
