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
from flask import Flask, jsonify
import time
import threading

ip="192.168.0.110"
app = Flask(__name__)
# 全局变量存储最新温湿度（避免频繁读取传感器导致卡顿）
latest_sensor_data = {
    "temperature": None,
    "humidity": None,
    "timestamp": None,  # 增加时间戳，方便排查
    "status": "normal"
}
dashscope.api_key = 'sk-ca23131efd1e4ceeb3812ce16097c37f'
# 配置信息
APPID = 'wx7564a41e542f83e9'
APPSECRET = '571989cddf2f9332e1952863aae2ef87'
openids = ["ojARd6nUw4tePDe1X80DWLY_oBv4", "ojARd6kPEAHuc0dy43CS6XWO7Qv8"]



warning="警告！未按规定佩戴防护措施！"
warning1=0
# def text_to_speech(text,warning1):
#     text="建议："+text
#     if warning1==1:
#         text=warning+text
#     result = SpeechSynthesizer.call(model='sambert-zhichu-v1',
#                                     text=text,
#                                     sample_rate=48000,
#                                     format='wav')
#     if result.get_audio_data() is not None:
#         with open('/home/wuqi/yolov8-prune/output.wav', 'wb') as f:
#             f.write(result.get_audio_data())
#         print('SUCCESS: get audio data: %dbytes in output.wav' %
#               (sys.getsizeof(result.get_audio_data())))
#     else:
#         print('ERROR: response is %s' % (result.get_response()))
#     subprocess.run(["play", "/home/wuqi/yolov8-prune/output.wav"])
#     print("said")

def dht(ser):
    # 读取串口一行数据
    dump = ser.readline()
    print(dump)
    # 转换为字符串（字节串转字符串后，\xc2\xb0会变成\\xc2\\xb0）
    dump = str(dump)
    # 清理字符串：移除开头的b'、末尾的'\r\n'等无关字符
    # 原始数据末尾是\r\n，转字符串后是\\r\\n，所以切片调整为[:-5]可能不准确，改用strip更通用
    dump = dump.strip("b'").strip("\\r\\n'")

    # 适配实际数据的正则表达式（匹配转义后的℃符号 \xc2\xb0C）
    # 匹配 Humidity: 27.00% 格式
    humidity_pattern = r"Humidity:\s+(\d+\.\d+)%"
    # 匹配 Temperature: 21.40\xc2\xb0C 格式（转字符串后是\\xc2\\xb0C）
    temperature_pattern = r"Temperature:\s+(\d+\.\d+)\\xc2\\xb0C"

    # 提取所有匹配的湿度和温度值（findall会返回全局所有匹配结果）
    humidities = re.findall(humidity_pattern, dump)
    temperatures = re.findall(temperature_pattern, dump)

    # 仅当同时获取到湿度和温度数据时返回（避免空列表）
    if humidities and temperatures:
        return humidities, temperatures


def text2openai(humi, temp,warning1):
    print(1)
    client = OpenAI(
        api_key="sk-a05fafcbfd6c43cdb9644035bf7b2987",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    str1 = "现在你是一名优秀的粮仓管理师，现在有一间面积为500平方米的存放玉米的粮仓，粮仓里有三个温度湿度传感器，分别位于粮仓的三个角落，每两个相距超过5米，现在第一个传回的温度是" + temp[
        0] + "，湿度是" + humi[0] + "，第二个传回的温度是" + temp[1] + "，湿度是" + humi[1] + "，第三个传回的温度是" + temp[
               2] + "，湿度是" + humi[2] + "，现在需要你用最简洁的语言告诉我这三个传感器,哪些传感器的温度湿度是否在正常范围内，以及需要哪些传感器做哪些调整（只需要回答温度湿度在正常内或不在正常范围内，并且给出不超过100字的简短建议）"
    completion = client.chat.completions.create(
        model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=[
            {'role': 'user', 'content': str1}
        ]
    )
    if warning1==1:
        print(warning)
    # 通过reasoning_content字段打印思考过程
    print("思考过程：")
    print(completion.choices[0].message.reasoning_content)
    # 通过content字段打印最终答案

    print("最终答案：")
    print(completion.choices[0].message.content)
    text=completion.choices[0].message.content
    # text_to_speech(text,warning1)
    # print("said")
    sender = WechatMessageSender(APPID, APPSECRET, openids)
    message_content ="这一个小时内:\n无人员未按规定佩戴防护措施!\n\n"+text

    # 调用 send_messages 方法发送消息
    sender.send_messages(message_content)
    # text_to_speech(completion.choices[0].message.content)

# def sensor_loop():
#     while (1):
#         try:
#             humi, temp = dht(ser)
#             print(temp)
#             print(humi)
#             # humi=[16.8,15.9,17.2]
#             # temp=[16.5,15.9,18.5]
#             text2openai(humi, temp,warning1)
#         except Exception as e:
#             print(f"Error in sensor loop: {e}")
#         time.sleep(1)  # 每10分钟（600秒）执行一次

def video_detection(warning1):
    # 加载 YOLO 模型
    model = YOLO("/home/wuqi/yolov8-prune/best.engine",verbose=False)

    # 打开视频文件
    cap = cv2.VideoCapture("finnal.mp4")  # 替换为你的视频文件路径

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error opening video file!")
        return

    # 初始化帧率和推理时间统计
    total_frames = 0
    processing_times = []  # 存储每帧的处理时间
    inference_times = []  # 存储每帧的推理时间
    start_time = cv2.getTickCount()

    # 初始化 FPS 和推理时间的显示
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有读取到帧，退出循环

        # 记录帧处理开始时间
        start_time_frame = cv2.getTickCount()

        # 使用 YOLO 模型进行预测
        start_time_inference = cv2.getTickCount()
        results = model.predict(frame)
        end_time_inference = cv2.getTickCount()
        inference_time = (end_time_inference - start_time_inference) / cv2.getTickFrequency() * 1000  # 毫秒
        inference_times.append(inference_time)

        # 遍历每个预测结果

        for result in results:
            boxes = result.boxes  # 获取边界框信息
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # 在视频帧上绘制矩形框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)
                if cls==1:
                    warning1=1
                    print(warning)

                # print(model.names[cls][0])


        # 记录帧处理结束时间
        end_time_frame = cv2.getTickCount()
        processing_time = (end_time_frame - start_time_frame) / cv2.getTickFrequency()  # 秒
        processing_times.append(processing_time)

        # 计算当前帧率和推理时间
        current_fps = round(1 / processing_time, 1) if processing_time > 0 else 0
        current_inference_time = round(inference_time, 2)

        # 在画面上显示 FPS 和推理时间
        cv2.putText(frame, f'FPS: {current_fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Inference Time: {current_inference_time} ms', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # 显示带有检测结果的视频帧
        cv2.imshow('YOLOv8', frame)
        # return warning1
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 程序结束后计算并输出平均 FPS 和平均推理时间7
    if len(processing_times) > 0 and len(inference_times) > 0:
        avg_fps = round(1 / (sum(processing_times) / len(processing_times)), 1)
        avg_inference_time = round(sum(inference_times) / len(inference_times), 2)
        print(f'平均 FPS: {avg_fps}')
        print(f'平均推理时间: {avg_inference_time} ms')
    else:
        print('No valid frames processed.')
    return warning1
def update_sensor_data_loop():
    """
    后台线程：持续更新传感器数据（每秒1次）
    避免API请求时才读取，导致响应慢
    """
    while True:
        try:
            humi, temp = dht(ser)
            # 更新全局变量
            latest_sensor_data["temperature"] = temp
            latest_sensor_data["humidity"] = humi
            latest_sensor_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            latest_sensor_data["status"] = "normal"
            text2openai(humi, temp, warning1)
        except Exception as e:
            # 捕获所有异常，避免线程崩溃
            latest_sensor_data["status"] = f"error: {str(e)}"
            print(f"❌ 读取传感器异常：{e}")
        # 读取间隔（可调整，比如5秒改5）
        time.sleep(1)


# ==================== Flask API接口定义 ====================
@app.route("/sensor", methods=["GET"])
def get_sensor_data():
    """
    核心API接口：GET请求访问/sensor即可获取温湿度数据
    返回格式：JSON（易解析）
    """
    return jsonify(latest_sensor_data)


@app.route("/health", methods=["GET"])
def health_check():
    """
    辅助接口：检查API服务是否正常运行
    """
    return jsonify({
        "status": "running",
        "jetson_ip": ip,  # 比如192.168.31.120
        "port": 5000
    })
if __name__ == "__main__":
    warning1 = 0
    ser = None  # 初始化串口变量
    try:
        # 1. 初始化串口
        ser = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=1)
        print("✅ 串口初始化成功")

        # 2. 创建传感器数据更新线程（设为守护线程，主线程退出时自动结束）
        sensor_thread = threading.Thread(target=update_sensor_data_loop, args=(), daemon=True)

        # 3. 创建YOLO视频检测线程（修正target参数，传入函数引用+参数）
        yolo_thread = threading.Thread(target=video_detection, args=(warning1,), daemon=True)

        # 4. 创建Flask Web服务线程（解决app.run阻塞问题）
        flask_thread = threading.Thread(target=app.run, args=(), kwargs={
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False,
            "use_reloader": False  # 关闭重载器，避免Flask启动多进程导致线程重复
        }, daemon=True)

        # 5. 启动所有线程（顺序不影响，均为并行执行）
        sensor_thread.start()
        print("✅ 传感器线程已启动")

        yolo_thread.start()
        print("✅ YOLO检测线程已启动")

        flask_thread.start()
        print("✅ Flask Web服务已启动 (http://192.168.0.110:5000)")

        # 6. 主线程保持运行（避免主线程退出导致所有守护线程终止）
        while True:
            time.sleep(1)  # 主线程持续等待

    except serial.SerialException as e:
        print(f"❌ 串口初始化失败：{e}")
    except Exception as e:
        print(f"❌ 程序启动异常：{e}")
    finally:
        # 程序退出时关闭串口
        if ser and ser.is_open:
            ser.close()
            print("✅ 串口已关闭")
        input("按任意键退出程序...")