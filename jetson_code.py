import os
import sys
from wxgzh import WechatMessageSender
from openai import OpenAI
from gtts import gTTS
import os
import serial
from serial import Serial
import re
import subprocess
import threading
import time
import cv2
from ultralytics import YOLO
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from flask import Flask, jsonify, Response
import numpy as np

# ==================== 全局配置 ====================
ip = "192.168.0.110"
app = Flask(__name__)

# 全局变量：温湿度数据
latest_sensor_data = {
    "temperature": None,
    "humidity": None,
    "timestamp": None,
    "status": "normal"
}

# 全局变量：防护措施警告（解决线程间参数传递问题）
global_warning1 = 0
warning = "警告！未按规定佩戴防护措施！"

# API密钥配置
dashscope.api_key = 'sk-ca23131efd1e4ceeb3812ce16097c37f'
openai_api_key = "sk-a05fafcbfd6c43cdb9644035bf7b2987"

# 微信配置
APPID = 'wx7564a41e542f83e9'
APPSECRET = '571989cddf2f9332e1952863aae2ef87'
openids = ["ojARd6nUw4tePDe1X80DWLY_oBv4", "ojARd6kPEAHuc0dy43CS6XWO7Qv8"]

# 摄像头配置（Jetson适配）
CAMERA_TYPE = "USB"  # 可选：CSI / USB
CAMERA_INDEX = 0  # USB默认0，CSI固定0
RESOLUTION = (640, 640)  # 分辨率，越小延迟越低
FPS = 30  # 帧率
JPEG_QUALITY = 80  # JPEG压缩质量（50-100）

# YOLO模型路径
YOLO_MODEL_PATH = "/home/wuqi/yolov8-prune/best.engine"


# ==================== 温湿度传感器读取 ====================
def dht(ser):
    """读取串口温湿度数据（适配原有逻辑）"""
    try:
        dump = ser.readline()
        if not dump:
            return None, None
        dump = str(dump)
        dump = dump.strip("b'").strip("\\r\\n'")

        # 正则提取温湿度（适配转义后的℃符号）
        humidity_pattern = r"Humidity:\s+(\d+\.\d+)%"
        temperature_pattern = r"Temperature:\s+(\d+\.\d+)\\xc2\\xb0C"

        humidities = re.findall(humidity_pattern, dump)
        temperatures = re.findall(temperature_pattern, dump)

        # 确保获取到3个传感器数据（粮仓3个角落）
        if len(humidities) >= 3 and len(temperatures) >= 3:
            return humidities[:3], temperatures[:3]
        else:
            print(f"⚠️ 传感器数据不完整：湿度{humidities}，温度{temperatures}")
            return None, None
    except Exception as e:
        print(f"❌ 温湿度解析失败：{e}")
        return None, None


# ==================== OpenAI分析 + 微信推送 ====================
# def text2openai(humi, temp):
#     """调用OpenAI分析温湿度，结合警告状态推送微信"""
#     global global_warning1
#     try:
#         client = OpenAI(
#             api_key=openai_api_key,
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#         )
#
#         # 构造提示词（适配3个传感器）
#         prompt = f"""现在你是一名优秀的粮仓管理师，管理500平方米玉米粮仓，有3个温湿度传感器（间距超5米）：
# 传感器1：温度{temp[0]}℃，湿度{humi[0]}%
# 传感器2：温度{temp[1]}℃，湿度{humi[1]}%
# 传感器3：温度{temp[2]}℃，湿度{humi[2]}%
# 请简洁说明各传感器温湿度是否正常，给出不超过100字的调整建议（仅说明正常/异常+建议）"""
#
#         completion = client.chat.completions.create(
#             model="deepseek-r1",
#             messages=[{"role": "user", "content": prompt}]
#         )
#
#         # 获取分析结果
#         analysis = completion.choices[0].message.content
#         print("📊 温湿度分析结果：", analysis)
#
#         # 构造微信消息（包含防护警告）
#         warning_msg = "有人员未按规定佩戴防护措施！\n\n" if global_warning1 == 1 else "无人员未按规定佩戴防护措施！\n\n"
#         message_content = f"这一个小时内:\n{warning_msg}{analysis}"
#
#         # 推送微信
#         sender = WechatMessageSender(APPID, APPSECRET, openids)
#         sender.send_messages(message_content)
#
#         return analysis
#     except Exception as e:
#         print(f"❌ OpenAI/微信推送失败：{e}")
#         return f"分析失败：{str(e)}"


# ==================== YOLO检测 + 视频流生成 ====================
def generate_yolo_video_stream():
    """
    核心函数：实时读取摄像头→YOLO检测→编码为JPEG流
    返回：视频流生成器（供Flask接口调用）
    """
    global global_warning1
    cap = None

    # 初始化摄像头（适配Jetson CSI/USB）
    try:
        if CAMERA_TYPE == "CSI":
            # Jetson CSI摄像头GStreamer管道（硬件加速）
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={CAMERA_INDEX} ! "
                f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, framerate={FPS}/1 ! "
                "nvvidconv flip-method=0 ! "
                "video/x-raw, format=BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! "
                "appsink drop=True"
            )
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        else:
            # USB摄像头配置
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            cap.set(cv2.CAP_PROP_FPS, FPS)

        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头！")
        print("✅ 摄像头初始化成功（类型：{}）".format(CAMERA_TYPE))

    except Exception as e:
        print(f"❌ 摄像头初始化失败：{e}")
        # 生成错误帧
        error_frame = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    # 加载YOLO模型
    try:
        model = YOLO(YOLO_MODEL_PATH, verbose=False)
        print("✅ YOLO模型加载成功")
    except Exception as e:
        print(f"❌ YOLO模型加载失败：{e}")
        cap.release()
        return

    # 实时处理帧并生成视频流
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 摄像头帧读取失败，重试...")
            time.sleep(0.1)
            continue

        # YOLO检测
        try:
            results = model.predict(frame, verbose=False)
            # 重置警告状态（每帧重新检测）
            frame_warning = 0

            # 绘制检测框 + 判断警告
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 绘制检测框和标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{model.names[cls]} {conf:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 检测到未佩戴防护措施（cls=1）
                    if cls == 1:
                        frame_warning = 1
                        global_warning1 = 1  # 更新全局警告状态
                        print(warning)

            # 无检测到违规则重置警告
            if frame_warning == 0 and global_warning1 == 1:
                global_warning1 = 0

            # 叠加警告信息和温湿度
            if global_warning1 == 1:
                cv2.putText(frame, warning, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 叠加温湿度（如果有数据）
            if latest_sensor_data["temperature"] and latest_sensor_data["humidity"]:
                temp_text = f"Temp: {latest_sensor_data['temperature'][0]}℃"
                humi_text = f"Humi: {latest_sensor_data['humidity'][0]}%"
                cv2.putText(frame, temp_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, humi_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # JPEG编码压缩
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # 生成视频流（符合HTTP流格式）
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"❌ 帧处理失败：{e}")
            continue

    # 释放资源（理论上不会执行到，除非循环终止）
    cap.release()


# ==================== 传感器数据更新线程 ====================
def update_sensor_data_loop(ser):
    """后台线程：持续读取温湿度并更新全局变量"""
    while True:
        try:
            humi, temp = dht(ser)
            if humi and temp:
                # 更新全局温湿度数据
                latest_sensor_data["temperature"] = temp
                latest_sensor_data["humidity"] = humi
                latest_sensor_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                latest_sensor_data["status"] = "normal"

                # 调用OpenAI分析（原逻辑：建议每小时执行，此处暂保留1秒间隔，可改600秒）
                text2openai(humi, temp)
            else:
                print("⚠️ 未获取到有效温湿度数据")
        except Exception as e:
            latest_sensor_data["status"] = f"error: {str(e)}"
            print(f"❌ 传感器线程异常：{e}")

        # 读取间隔（原1秒，粮仓建议改为3600秒=1小时）
        time.sleep(1)


# ==================== Flask接口 ====================
@app.route("/sensor", methods=["GET"])
def get_sensor_data():
    """温湿度数据接口"""
    return jsonify(latest_sensor_data)


@app.route("/health", methods=["GET"])
def health_check():
    """服务健康检查接口"""
    return jsonify({
        "status": "running",
        "jetson_ip": ip,
        "port": 5000,
        "warning_status": "违规" if global_warning1 == 1 else "正常",
        "camera_type": CAMERA_TYPE
    })


@app.route("/video_feed")
def video_feed():
    """YOLO检测后的视频流接口"""
    return Response(
        generate_yolo_video_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ==================== 主程序 ====================
if __name__ == "__main__":
    ser = None
    try:
        # 1. 初始化串口
        ser = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=1)
        print("✅ 串口初始化成功")

        # 2. 启动传感器数据更新线程
        sensor_thread = threading.Thread(
            target=update_sensor_data_loop,
            args=(ser,),
            daemon=True
        )
        sensor_thread.start()
        print("✅ 传感器线程已启动")

        # 3. 启动Flask服务（包含视频流+温湿度接口）
        print(f"✅ Flask服务启动中... (http://{ip}:5000)")
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False  # 关键：关闭重载器，避免多进程冲突
        )

    except serial.SerialException as e:
        print(f"❌ 串口初始化失败：{e}")
    except Exception as e:
        print(f"❌ 程序启动异常：{e}")
    finally:
        # 程序退出时关闭串口
        if ser and ser.is_open:
            ser.close()
            print("✅ 串口已关闭")
        # 释放摄像头（如果有）
        cv2.destroyAllWindows()
        print("✅ 程序正常退出")