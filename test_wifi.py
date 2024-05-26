import requests
import cv2
import numpy as np

# Địa chỉ IP của ESP32, thay đổi '192.168.x.x' thành địa chỉ IP của ESP32 của bạn
esp32_ip = 'http://10.10.59.136:81'

# URL để lấy stream video
stream_url = f'{esp32_ip}:81/stream'

# Kết nối tới stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Không thể kết nối tới stream video")
    exit()

while True:
    # Đọc frame từ stream
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận frame")
        break

    # Hiển thị frame
    cv2.imshow('ESP32 Camera Stream', frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
