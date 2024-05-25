import os

server_ip = os.getenv('SERVER_IP', '127.0.0.1')
api_url = f'http://{server_ip}:5000/account-management/account/check'

# Tiếp tục sử dụng `api_url` trong mã nguồn của bạn
