import requests
# Địa chỉ URL của Flask server

url = 'http://10.10.58.180:5000/history-management/history'

# Dữ liệu bạn muốn gửi
data = {
    'vehicle_plate': '75H12917',
}

# Gửi yêu cầu POST đến server
response = requests.post(url, json=data)

# Kiểm tra kết quả
if response.status_code == 200:
    print("Dữ liệu đã được gửi thành công!")
else:
    print("Đã xảy ra lỗi:", response.status_code)