import cv2
import requests
import numpy as np

JETSON_IP = "192.168.0.110"
VIDEO_URL = f"http://{JETSON_IP}:5000/video_feed"

# 读取视频流
cap = cv2.VideoCapture(VIDEO_URL)
if not cap.isOpened():
    print("❌ 无法连接到视频流")
    exit()

print("✅ 视频流已连接，按'q'退出")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Jetson YOLO Video Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()