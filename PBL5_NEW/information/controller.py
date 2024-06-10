from flask import Blueprint
from .services import (
    add_information_service,
    get_information_by_cccd_service,
    get_all_informations_service,
    update_information_by_cccd_service,
    delete_information_by_cccd_service,
    get_information_by_id_account_service,
    delete_information_by_id_account_service,
    update_information_service
)

information = Blueprint("information", __name__)

# Add new information
@information.route("/information-management/information", methods=['POST'])
def add_information():
    return add_information_service()

# Get information by cccd
@information.route("/information-management/information/<string:cccd>", methods=['GET'])
def get_information_by_cccd(cccd):
    return get_information_by_cccd_service(cccd)

# Get all informations
@information.route("/information-management/informations", methods=['GET'])
def get_all_informations():
    return get_all_informations_service()

# Update information
@information.route("/information-management/information/<string:cccd>", methods=['PUT'])
def update_information_by_cccd(cccd):
    return update_information_by_cccd_service(cccd)

# Delete information
@information.route("/information-management/information/<string:cccd>", methods=['DELETE'])
def delete_information_by_cccd(cccd):
    return delete_information_by_cccd_service(cccd)

# Get information by id_account
@information.route("/information-management/information/account/<int:id_account>", methods=['GET'])
def get_information_by_id_account(id_account):
    return get_information_by_id_account_service(id_account)

# Delete information by id_account
@information.route("/information-management/information/account/<int:id_account>", methods=['DELETE'])
def delete_information_by_id_account(id_account):
    return delete_information_by_id_account_service(id_account)

@information.route('/information-management/information/account/<int:id_account>', methods=['PUT'])
def update_information(id_account):
    return update_information_service(id_account)