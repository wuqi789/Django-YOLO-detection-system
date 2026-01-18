import requests
import time
import json

# ==================== 需修改的配置项 ====================
JETSON_IP = "192.168.0.110"  # 替换为你的Jetson真实IP
JETSON_PORT = 5000  # 必须和Jetson端的port一致
# =======================================================

# 拼接API地址
SENSOR_API_URL = f"http://{JETSON_IP}:{JETSON_PORT}/sensor"
HEALTH_API_URL = f"http://{JETSON_IP}:{JETSON_PORT}/health"


def check_jetson_health():
    """先检查Jetson的API服务是否可达"""
    try:
        response = requests.get(HEALTH_API_URL, timeout=3)
        if response.status_code == 200:
            print("✅ Jetson API服务正常运行")
            return True
        else:
            print(f"❌ Jetson API服务返回异常状态码：{response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请检查：")
        print("  1. Jetson和电脑是否在同一局域网")
        print("  2. Jetson的Flask服务是否已启动")
        print("  3. IP和端口是否填写正确")
        print("  4. Jetson防火墙是否关闭（Jetson默认关闭）")
        return False
    except Exception as e:
        print(f"❌ 检查健康状态异常：{e}")
        return False


def get_sensor_data_continuously():
    """持续获取温湿度数据（每秒1次）"""
    # 先检查健康状态
    if not check_jetson_health():
        return

    print("\n开始接收温湿度数据（按Ctrl+C停止）：")
    print("-" * 50)

    while True:
        try:
            # 发送GET请求获取数据
            response = requests.get(SENSOR_API_URL, timeout=5)

            # 检查响应状态
            if response.status_code == 200:
                # 解析JSON数据
                data = response.json()
                # 打印格式化数据
                print(
                    f"时间：{data['timestamp']} | 温度：{data['temperature']}℃ | 湿度：{data['humidity']}% | 状态：{data['status']}")
                # 格式为:时间：2026-01-18 19:38:32 | 温度：['21.80', '19.60', '19.70']℃ | 湿度：['29.00', '46.50', '46.60']% | 状态：normal
            else:
                print(f"❌ 获取数据失败，状态码：{response.status_code}")

        except json.JSONDecodeError:
            print("❌ 数据格式错误，非JSON格式")
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络请求异常：{e}")
            # 异常后重试前等待3秒
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n✅ 停止接收数据")
            break
        except Exception as e:
            print(f"❌ 未知异常：{e}")

        # 数据获取间隔（和Jetson端一致即可）
        time.sleep(1)


if __name__ == "__main__":
    get_sensor_data_continuously()
