from flask import request, jsonify
from PBL5.extension import db
from PBL5.pbl5_ma import AccountSchema
from PBL5.model import Account
import json

account_schema = AccountSchema()
accounts_schema = AccountSchema(many=True)

def add_account_service():
    data = request.json
    if data and 'username' in data and 'password' in data and 'name' in data and 'phone' in data:
        username = data['username']
        password = data['password']
        name = data['name']
        phone = data['phone']
        try:
            new_account = Account(username=username, password=password, name=name, phone=phone)
            db.session.add(new_account)
            db.session.commit()
            return jsonify({"message": "Account added successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")  # In ra lỗi để dễ dàng kiểm tra
            return jsonify({"message": "Could not add account!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Request error: Missing fields"}), 400

def get_account_by_id_service(id):
    account = Account.query.get(id)
    if account:
        return account_schema.jsonify(account)
    else:
        return jsonify({"message": "Account not found"}), 404

def get_all_accounts_service():
    accounts = Account.query.all()
    return accounts_schema.jsonify(accounts)

def update_account_by_id_service(id):
    account = Account.query.get(id)
    data = request.json
    if account:
        if data and 'username' in data:
            try:
                account.username = data['username']
                account.password = data['password']
                account.name = data['name']
                account.phone = data['phone']
                db.session.commit()
                return jsonify({"message": "Account updated successfully!"}), 200
            except Exception as e:
                db.session.rollback()
                return jsonify({"message": "Could not update account!"}), 400
    else:
        return jsonify({"message": "Account not found"}), 404

def delete_account_by_id_service(id):
    account = Account.query.get(id)
    if account:
        try:
            db.session.delete(account)
            db.session.commit()
            return jsonify({"message": "Account deleted successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"message": "Could not delete account!"}), 400
    else:
        return jsonify({"message": "Account not found"}), 404

def check_account_service():
    data = request.json
    account = Account.query.filter_by(username=data['username']).first()
    if account is None:
        return jsonify({"message": "Account not found"}),400
    else:
        if account.password != data['password']:
            return jsonify({"message": "Password is incorrect"}),400
        else:
            return jsonify({"message": "Login successful"});200