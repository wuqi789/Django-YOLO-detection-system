import requests
import json
import time

class WechatMessageSender:
    def __init__(self, appid, appsecret, openids):
        self.APPID = appid
        self.APPSECRET = appsecret
        self.openids = openids
        # 全局变量用于存储 Access Token
        # 在生产环境中，应该使用更健壮的方式存储和管理 access_token，
        # 例如 Redis 或数据库，并记录过期时间。
        self.access_token_info = {
            'token': None,
            'expires_at': 0  # 过期时间戳
        }

    def get_access_token(self):
        """获取或刷新 Access Token"""
        now = time.time()

        # 检查 token 是否有效 (留有一定冗余时间，比如提前 5 分钟刷新)
        if self.access_token_info['token'] and self.access_token_info['expires_at'] > now + 300:
            print("使用缓存的 Access Token")
            return self.access_token_info['token']

        print("获取新的 Access Token...")
        url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={self.APPID}&secret={self.APPSECRET}"
        try:
            response = requests.get(url, timeout=5)  # 设置超时
            response.raise_for_status()  # 如果请求失败则抛出 HTTPError 异常
            result = response.json()

            if 'access_token' in result:
                self.access_token_info['token'] = result['access_token']
                # 微信返回的 expires_in 是秒数，计算出绝对过期时间戳
                self.access_token_info['expires_at'] = now + result.get('expires_in', 7200)
                print(
                    f"获取成功, Token: {self.access_token_info['token'][:10]}..., 将在 {result.get('expires_in', 7200)} 秒后过期")
                return self.access_token_info['token']
            else:
                print(f"获取 Access Token 失败: {result}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"请求 Access Token 时发生错误: {e}")
            return None
        except json.JSONDecodeError:
            print("解析 Access Token 响应失败")
            return None

    def send_text_message(self, openid, message_content):
        """使用客服消息接口发送文本消息"""
        token = self.get_access_token()
        print(token)
        if not token:
            print("无法获取 Access Token，发送失败")
            return False

        url = f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={token}"

        payload = {
            "touser": openid,
            "msgtype": "text",
            "text": {
                "content": message_content
            }
        }

        headers = {'Content-Type': 'application/json; charset=utf-8'}

        try:
            # 手动序列化并指定编码
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            response = requests.post(url, data=data, headers=headers, timeout=5)
            response.raise_for_status()
            result = response.json()

            if result.get('errcode') == 0:
                print(f"消息成功发送给 {openid}")
                return True
            else:
                print(f"消息发送失败: {result}")
                # 特别处理 48 小时限制错误
                if result.get('errcode') == 45015:
                    print("错误详情：该用户可能超过48小时未与公众号互动")
                # 特别处理 token 失效错误 (虽然 get_access_token 应该处理了，但以防万一)
                elif result.get('errcode') in [40001, 40014, 42001]:
                    print("错误详情：Access Token 无效或已过期，请尝试重新运行程序获取新 Token。")
                    # 可以考虑强制清空缓存的 token
                    self.access_token_info['token'] = None
                    self.access_token_info['expires_at'] = 0
                return False
        except requests.exceptions.RequestException as e:
            print(f"发送消息时发生网络错误: {e}")
            return False
        except json.JSONDecodeError:
            print("解析发送消息响应失败")
            return False

    def send_messages(self, message_content):
        all_success = True
        for openid in self.openids:
            success = self.send_text_message(openid, message_content)
            if not success:
                all_success = False
        if all_success:
            print("任务完成。")
        else:
            print("任务失败。")


# --- 主程序 ---
if __name__ == "__main__":
    # 配置信息
    APPID = 'wx7564a41e542f83e9'
    APPSECRET = '571989cddf2f9332e1952863aae2ef87'
    # 指定接收消息的粉丝的 OpenID
    # !!! 重要: 这些 OpenID 需要你通过其他方式获取并存储 !!!
    # 例如，用户关注或发消息给公众号时，你的后台服务可以记录下来。
    openids = ["ojARd6nUw4tePDe1X80DWLY_oBv4", "ojARd6kPEAHuc0dy43CS6XWO7Qv8"]

    sender = WechatMessageSender(APPID, APPSECRET, openids)
    # 1. 定义你要发送的字符串
    my_string_message = "你好！这是一条来自 Python 脚本的测试消息。"
    sender.send_messages(my_string_message)
