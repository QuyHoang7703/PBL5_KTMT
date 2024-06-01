from flask import Blueprint,jsonify
from .services import (add_history_service, get_history_by_id_service,
                       get_all_histories_service, update_history_by_id_service,
                       delete_history_by_id_service)
histories = Blueprint("histories", __name__)
import time
import threading
last_request_time = 0
lock = threading.Lock()  # Khai báo biến lock ở phạm vi toàn cục
# Add a new history
@histories.route("/history-management/history", methods=['POST'])
def add_history():
    global last_request_time
    global lock  # Sử dụng biến lock đã được khai báo ở phạm vi toàn cục
    
    with lock:
        current_time = time.time()
        
        # Kiểm tra xem đã đủ 30 giây kể từ lần yêu cầu cuối chưa
        if current_time - last_request_time < 30:
            return jsonify(error="Đợi 30 giây trước khi gửi yêu cầu mới"), 429
        
        # Cập nhật thời gian của yêu cầu cuối cùng
        last_request_time = current_time
        
    return add_history_service()

# Get history by id
@histories.route("/history-management/history/<int:id>", methods=['GET'])
def get_history_by_id(id):
    return get_history_by_id_service(id)

# Get all histories
@histories.route("/history-management/histories", methods=['GET'])
def get_all_histories():
    return get_all_histories_service()

# Update history
@histories.route("/history-management/history/<int:id>", methods=['PUT'])
def update_history_by_id(id):
    return update_history_by_id_service(id)

# Delete history
@histories.route("/history-management/history/<int:id>", methods=['DELETE'])
def delete_history_by_id(id):
    return delete_history_by_id_service(id)
