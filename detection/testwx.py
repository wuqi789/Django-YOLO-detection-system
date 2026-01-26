from wxgzh import WechatMessageSender

# 配置信息
APPID = 'wx7564a41e542f83e9'
APPSECRET = '571989cddf2f9332e1952863aae2ef87'
openids = ["ojARd6nUw4tePDe1X80DWLY_oBv4"]

# 创建 WechatMessageSender 实例
sender = WechatMessageSender(APPID, APPSECRET, openids)

# 定义要发送的字符串消息
import requests,json
# access_token='90_2iYfuRrzDcf4sRHSqtE8xsc8CNMbcUsgWwS6mIp570YESnDItRage0-jLfc_uNg2D8SOLipGbhuoI5n4ps0CPsCY7OEYaJgsChuu_GLQ2kS3DPLwBXqT_sy33hIRNCbAHAIET'
# import requests,json
# # access_token='相关公众号的token' #未认证的订阅号不能获取，也就是没有权限获取粉丝openid
# next_openid=''
# url='https://api.weixin.qq.com/cgi-bin/user/get?access_token=%s&next_openid=%s'%(access_token,next_openid)
# ans=requests.get(url)
# #print(ans.content)
# a=json.loads(ans.content)['data']['openid']
# print (a,len(a))
message_content = "你好！这是一条来自 Python 脚本的测试消息。"# 调用 send_messages 方法发送消息
sender.send_messages(message_content)