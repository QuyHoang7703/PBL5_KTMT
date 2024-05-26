from flask import Flask, request
import requests  # Import thư viện requests

app = Flask(__name__)
previous_license_plate = None
# url_to_django = 'http://127.0.0.1:8000/'
url = 'http://10.10.58.180:5000/history-management/history'
@app.route('/api/receive_send', methods=['POST'])
def receive_send():
    global previous_license_plate
    data = request.json
    if 'license_plate' in data:
        license_plate = data['license_plate']
        if license_plate != previous_license_plate:
            previous_license_plate = license_plate
            print("Received new license plate:", license_plate)

            # Gửi request tới django url
            # try:
            #     response = requests.post(url_to_django , json=data)
            #     response.raise_for_status()  # Kiểm tra lỗi
            #     print("Request sent successfully")
            # except requests.exceptions.RequestException as e:
            #     print("Error sending request:", e)
            # Gửi yêu cầu POST đến server
            response = requests.post(url, json=data)

            # Kiểm tra kết quả
            if response.status_code == 200:
                print("Dữ liệu đã được gửi thành công!")
            else:
                print("Đã xảy ra lỗi:", response.status_code)
            return "New data received, updated, and request sent"
        else:
            print("Received duplicate license plate:", license_plate)
            return "Duplicate data received, no update needed"
    else:
        return "License plate not found in data"

if __name__ == '__main__':
    app.run(debug=True)