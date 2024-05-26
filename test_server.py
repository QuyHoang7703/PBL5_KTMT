from flask import Flask, request

app = Flask(__name__)

# Khởi tạo một set để lưu trữ các biển số xe
license_plates_set = set()


@app.route('/api/receive_send', methods=['POST'])
def receive_send():
    data = request.json
    if 'license_plate' in data:
        license_plate = data['license_plate']
        print("Received license plate:", license_plate)

        # Kiểm tra xem biển số xe đã tồn tại trong set chưa
        if license_plate not in license_plates_set:
            # Nếu chưa tồn tại, xoá hết các giá trị cũ trong set và thêm vào biển số xe mới
            license_plates_set.clear()
            license_plates_set.add(license_plate)
            print("License plate added to set.")

            # In ra tất cả các biển số xe trong set
            print("Current set:", license_plates_set)

            # Thực hiện các xử lý khác với biển số xe nhận được ở đây

            return "Data received successfully."
        else:
            print("License plate already exists in set.")
            return "License plate already exists.", 400
    else:
        return "Invalid data format.", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)