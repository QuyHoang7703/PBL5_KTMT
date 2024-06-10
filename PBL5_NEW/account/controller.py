from flask import Blueprint , session ,request
from .services import (
    add_account_service,
    get_account_by_id_service,
    get_all_accounts_service,
    update_account_by_id_service,
    delete_account_by_id_service,
    check_account_service,
    check_username_exists_service,
    change_password_service
)

accounts = Blueprint("accounts", __name__)

# Add a new account
@accounts.route("/account-management/account", methods=['POST'])
def add_account():
    return add_account_service()

# Get account by id
@accounts.route("/account-management/account/<int:id>", methods=['GET'])
def get_account_by_id(id):
    return get_account_by_id_service(id)

# Get all accounts
@accounts.route("/account-management/accounts", methods=['GET'])
def get_all_accounts():
    return get_all_accounts_service()

# Update account
@accounts.route("/account-management/account/<int:id>", methods=['PUT'])
def update_account_by_id(id):
    return update_account_by_id_service(id)

# Delete account
@accounts.route("/account-management/account/<int:id>", methods=['DELETE'])
def delete_account_by_id(id):
    return delete_account_by_id_service(id)

# Check account
@accounts.route("/account-management/account/check", methods=['POST'])
def check_account():
    data = request.json
    session['id_account'] = data['username']
    return check_account_service()

# Check if username exists
@accounts.route("/account-management/account/check-username/<string:username>", methods=['GET'])
def check_username_exists(username):
    return check_username_exists_service(username)

# Change password
@accounts.route("/account-management/account/change-password", methods=['POST'])
def change_password():
    return change_password_service()
