from flask import request, jsonify
from PBL5_NEW.extension import db
from PBL5_NEW.pbl5_ma import HistorySchema
from PBL5_NEW.model import History , Information
import json
from datetime import datetime
import requests

history_schema = HistorySchema()
histories_schema = HistorySchema(many=True)

from datetime import datetime
from PBL5_NEW.model import Ticket

def add_history_service():
    data = request.json
    vehicle_plate = data.get('vehicle_plate')

    if not vehicle_plate:
        return jsonify({'error': 'Vehicle plate is required'}), 400

    # Kiểm tra nếu biển số xe có trong bảng Ticket
    ticket = Ticket.query.filter_by(vehicle_plate=vehicle_plate).first()

    if not ticket:
        return jsonify({'error': 'Vehicle plate not found in tickets'}), 404

    # Kiểm tra nếu ticket.status bằng 0 và expiry lớn hơn ngày hôm nay
    if ticket.status != 0:
        return jsonify({'error': 'Ticket is not approved'}), 400

    if ticket.expiry < datetime.now().date():
        return jsonify({'error': 'Ticket has expired'}), 400

    # Lấy ra biển số xe của lịch sử gần nhất (nếu có)
    latest_history = History.query.order_by(History.date_in.desc(), History.time_in.desc()).first()

    history = History.query.filter_by(vehicle_plate=vehicle_plate).order_by(History.date_in.desc(), History.time_in.desc()).first()
    if history and not history.date_out and not history.time_out:
        # Nếu đã có ngày giờ vào mà chưa có ngày giờ ra, cập nhật ngày giờ ra
        history.date_out = datetime.strptime(data.get('date'), '%d-%m-%Y').date()
        history.time_out = datetime.strptime(data.get('time'), '%H:%M:%S').time()

        db.session.commit()
        esp8266_ip = "http://192.168.234.150/open_door_ra"
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

        esp8266_ip = "http://192.168.234.150/open_door_vao"
        response = requests.get(esp8266_ip)

        return jsonify({'message': 'New history entry created with check-in time'}), 200

# Biến để lưu trữ trạng thái của yêu cầu
# last_request_type = None
# last_vehicle_in = None
# last_vehicle_out = None

# def add_history_service():
#     global last_request_type, last_vehicle_in, last_vehicle_out

#     data = request.json
#     vehicle_plate = data.get('vehicle_plate')
#     request_type = data.get('request_type')  # 'in' for check-in, 'out' for check-out

#     if not vehicle_plate or not request_type:
#         return jsonify({'error': 'Vehicle plate and request type are required'}), 400

#     # Kiểm tra nếu biển số xe có trong bảng Ticket
#     ticket = Ticket.query.filter_by(vehicle_plate=vehicle_plate).first()

#     if not ticket:
#         return jsonify({'error': 'Vehicle plate not found in tickets'}), 404

#     # Kiểm tra nếu ticket.status bằng 0 và expiry lớn hơn ngày hôm nay
#     if ticket.status != 0:
#         return jsonify({'error': 'Ticket is not approved'}), 400

#     if ticket.expiry < datetime.now().date():
#         return jsonify({'error': 'Ticket has expired'}), 400

#     # Kiểm tra yêu cầu trùng lặp
#     if request_type == 'in' and last_request_type == 'in' and last_vehicle_in == vehicle_plate:
#         return jsonify({'error': 'Duplicate check-in request detected'}), 400
#     elif request_type == 'out' and last_request_type == 'out' and last_vehicle_out == vehicle_plate:
#         return jsonify({'error': 'Duplicate check-out request detected'}), 400

#     history = History.query.filter_by(vehicle_plate=vehicle_plate).order_by(History.date_in.desc(), History.time_in.desc()).first()
#     if history and not history.date_out and not history.time_out:
#         if request_type == 'out':
#             # Nếu đã có ngày giờ vào mà chưa có ngày giờ ra, cập nhật ngày giờ ra
#             history.date_out = datetime.strptime(data.get('date'), '%d-%m-%Y').date()
#             history.time_out = datetime.strptime(data.get('time'), '%H:%M:%S').time()

#             db.session.commit()

#             # Cập nhật trạng thái và biển số xe cuối cùng cho cửa ra
#             last_request_type = 'out'
#             last_vehicle_out = vehicle_plate

#             return jsonify({'message': 'History updated with check-out time'}), 200
#         else:
#             return jsonify({'error': 'Vehicle already checked in'}), 400
#     else:
#         if request_type == 'in':
#             # Nếu chưa có hoặc đã có ngày giờ ra, tạo một lịch sử mới với ngày giờ vào
#             new_history = History(
#                 vehicle_plate=vehicle_plate,
#                 date_in=datetime.strptime(data.get('date'), '%d-%m-%Y').date(),
#                 time_in=datetime.strptime(data.get('time'), '%H:%M:%S').time(),
#                 date_out=None,
#                 time_out=None
#             )
#             db.session.add(new_history)
#             db.session.commit()

#             # Cập nhật trạng thái và biển số xe cuối cùng cho cửa vào
#             last_request_type = 'in'
#             last_vehicle_in = vehicle_plate

#             # Gửi tín hiệu đến ESP8266 để mở cửa
#             # esp8266_ip = "http://10.10.49.204/open_door"
#             # requests.get(esp8266_ip)

#             return jsonify({'message': 'New history entry created with check-in time'}), 200
#         else:
#             return jsonify({'error': 'Vehicle not checked in yet'}), 400
        
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

def get_histories_by_id_customer_service(id_customer):
    # Lấy thông tin khách hàng dựa trên id_customer
    information = Information.query.filter_by(id_account=id_customer).first()
    if not information:
        return jsonify({"message": "Customer not found"}), 404

    # Lấy cccd từ thông tin khách hàng
    cccd = information.cccd

    # Truy vấn tất cả các ticket liên quan đến cccd
    tickets = Ticket.query.filter_by(cccd=cccd).all()
    if not tickets:
        return jsonify({"message": "No tickets found for the provided customer ID"}), 404

    # Lấy tất cả vehicle_plate từ các tickets
    vehicle_plates = [ticket.vehicle_plate for ticket in tickets]

    # Truy vấn tất cả các lịch sử liên quan đến vehicle_plates
    histories = History.query.filter(History.vehicle_plate.in_(vehicle_plates)).all()
    return histories_schema.jsonify(histories), 200

from sqlalchemy import and_

def get_filtered_histories(customer_id, date_in, date_out, time_in, time_out):
    query = db.session.query(History).join(Ticket).join(Information).filter(Information.id_account == customer_id)
    
    if date_in:
        query = query.filter(History.date_in >= date_in)
    if date_out:
        query = query.filter(History.date_out <= date_out)
    if time_in:
        query = query.filter(History.time_in >= time_in)
    if time_out:
        query = query.filter(History.time_out <= time_out)
        
    histories = query.all()
    return [history_to_dict(history) for history in histories]

def history_to_dict(history):
    return {
        'id_history': history.id_history,
        'vehicle_plate': history.vehicle_plate,
        'date_in': history.date_in.isoformat() if history.date_in else None,
        'date_out': history.date_out.isoformat() if history.date_out else None,
        'time_in': history.time_in.isoformat() if history.time_in else None,
        'time_out': history.time_out.isoformat() if history.time_out else None
    }

def get_filtered_histories_admin(date_in=None, date_out=None, time_in=None, time_out=None):
    query = db.session.query(History)
    
    if date_in:
        query = query.filter(History.date_in >= date_in)
    if date_out:
        query = query.filter(History.date_out <= date_out)
    if time_in:
        query = query.filter(History.time_in >= time_in)
    if time_out:
        query = query.filter(History.time_out <= time_out)
        
    histories = query.all()
    return [history_to_dict(history) for history in histories]