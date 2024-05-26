from flask import Flask, request
import requests
import threading
import logging

app = Flask(__name__)

# Khởi tạo một set để lưu trữ các biển số xe
license_plates_set = set()
lock = threading.Lock()

# Thiết lập logging
logging.basicConfig(level=logging.INFO)

@app.route('/api/receive_send', methods=['POST'])
def receive_send():
    data = request.json
    if 'license_plate' in data:
        license_plate = data['license_plate']
        logging.info(f"Received license plate: {license_plate}")

        with lock:  # Dùng khóa để bảo vệ đoạn mã truy cập vào license_plates_set
            if license_plate not in license_plates_set:
                # Nếu chưa tồn tại, xoá hết các giá trị cũ trong set và thêm vào biển số xe mới
                license_plates_set.clear()
                license_plates_set.add(license_plate)
                logging.info("License plate added to set.")

                # In ra tất cả các biển số xe trong set
                logging.info(f"Current set: {license_plates_set}")

                # Lấy một giá trị từ set và gửi lên Django
                django_url = 'http://127.0.0.1:8000/'
                try:
                    license_plate_to_send = next(iter(license_plates_set))
                    response = requests.post(django_url, json=license_plate_to_send)
                    response.raise_for_status()
                    logging.info("Data sent to Django successfully.")
                except (StopIteration, requests.RequestException) as e:
                    logging.error(f"Failed to send data to Django: {e}")
                    return "Failed to send data to Django.", 500

                return "Data received successfully."
            else:
                logging.info("License plate already exists in set.")
                return "License plate already exists.", 400
    else:
        return "Invalid data format.", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
