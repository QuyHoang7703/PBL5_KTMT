from flask import Blueprint
from .services import (add_history_service, get_history_by_id_service,
                       get_all_histories_service, update_history_by_id_service,
                       delete_history_by_id_service)
histories = Blueprint("histories", __name__)

# Add a new history
@histories.route("/history-management/history", methods=['POST'])
def add_history():
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
