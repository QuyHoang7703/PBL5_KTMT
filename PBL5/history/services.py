from flask import request, jsonify
from PBL5.extension import db
from PBL5.pbl5_ma import HistorySchema
from PBL5.model import History
import json
from datetime import datetime

history_schema = HistorySchema()
histories_schema = HistorySchema(many=True)

from datetime import datetime
from PBL5.model import Ticket
import requests , time
def add_history_service():
    data = request.json
    vehicle_plate = data.get('vehicle_plate')

    if not vehicle_plate:
        return jsonify({'error': 'Vehicle plate is required'}), 400

    # Kiểm tra nếu biển số xe có trong bảng Ticket
    ticket = Ticket.query.filter_by(vehicle_plate=vehicle_plate).first()

    if not ticket:
        return jsonify({'error': 'Vehicle plate not found in tickets'}), 404

    # Lấy ra biển số xe của lịch sử gần nhất (nếu có)
    latest_history = History.query.order_by(History.date_in.desc(), History.time_in.desc()).first()

    # if latest_history and latest_history.vehicle_plate == vehicle_plate:
    #     return jsonify({'message': 'Vehicle already checked in'}), 200
    # else:
    history = History.query.filter_by(vehicle_plate=vehicle_plate).order_by(History.date_in.desc(), History.time_in.desc()).first()
    if history and not history.date_out and not history.time_out:
            # Nếu đã có ngày giờ vào mà chưa có ngày giờ ra, cập nhật ngày giờ ra
            # history.date_out = datetime.now().date()
            # history.time_out = datetime.now().time()
        history.date_out = datetime.strptime(data.get('date'), '%d-%m-%Y').date()
        history.time_out = datetime.strptime(data.get('time'), '%H:%M:%S').time()

        db.session.commit()
        esp8266_ip = "http://192.168.174.150/open_door_ra"
        response = requests.get(esp8266_ip)
        return jsonify({'message': 'History updated with check-out time'}), 200
    else:
        # Nếu chưa có hoặc đã có ngày giờ ra, tạo một lịch sử mới với ngày giờ vào
        new_history = History(
            vehicle_plate=vehicle_plate,
            date_in=datetime.strptime(data.get('date'), '%d-%m-%Y').date(),
            time_in=datetime.strptime(data.get('time'), '%H:%M:%S').time(),
            date_out=None,
            time_out=None
        )
        db.session.add(new_history)
        db.session.commit()
        esp8266_ip = 'http://192.168.174.150/open_door_vao'
        response = requests.get(esp8266_ip)
        # retries = 5
        # delay = 2  # Thời gian chờ ban đầu tính bằng giây
        #
        # for i in range(retries):
        #     try:
        #         response = requests.get(esp8266_ip, timeout=10)
        #         response.raise_for_status()  # Tạo lỗi HTTPError cho các phản hồi xấu
        #         break  # Nếu yêu cầu thành công, thoát vòng lặp
        #     except requests.exceptions.RequestException as e:
        #         print(f"Thử lần {i + 1} thất bại: {e}")
        #         time.sleep(delay)
        #         delay *= 2  # Backoff theo cấp số nhân
        # else:
        #     print("Tất cả các lần thử kết nối đến máy chủ đều thất bại.")

        return jsonify({'message': 'New history entry created with check-in time'}), 200

def get_history_by_id_service(id):
    history = History.query.get(id)
    if history:
        return history_schema.jsonify(history)
    else:
        return jsonify({"message": "History not found"}), 404

def get_all_histories_service():
    histories = History.query.all()
    return histories_schema.jsonify(histories)

def update_history_by_id_service(id):
    history = History.query.get(id)
    data = request.json
    if history:
        if data and 'vehicle_plate' in data:
            try:
                history.vehicle_plate = data['vehicle_plate']
                history.date_in = data['date_in']
                history.date_out = data['date_out']
                history.time_in = data['time_in']
                history.time_out = data['time_out']
                db.session.commit()
                return jsonify({"message": "History updated successfully!"}), 200
            except Exception as e:
                db.session.rollback()
                return jsonify({"message": "Could not update history!"}), 400
    else:
        return jsonify({"message": "History not found"}), 404

def delete_history_by_id_service(id):
    history = History.query.get(id)
    if history:
        try:
            db.session.delete(history)
            db.session.commit()
            return jsonify({"message": "History deleted successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"message": "Could not delete history!"}), 400
    else:
        return jsonify({"message": "History not found"}), 404
