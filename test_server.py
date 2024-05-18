from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


@app.route('/api/receive', methods=['POST'])
def receive_send_data():
    try:
        # Nhận dữ liệu từ request của Flask
        data = request.get_json()
        license_plate = data.get('license_plate', None)
        print(f"Received license plate: {license_plate}")

        # URL của endpoint trong Django
        django_url = "http://127.0.0.1:8000/"

        # Dữ liệu JSON bạn muốn gửi đến Django
        data_to_send = {'license_plate': license_plate}

        # Gửi yêu cầu POST tới Django với dữ liệu JSON
        response = requests.post(django_url, json=data_to_send)

        # In ra kết quả từ Django
        print("Response from Django:", response.text)

        return jsonify({"status": "success", "message": "Data sent to Django successfully"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
