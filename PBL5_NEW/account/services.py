from flask import request, jsonify,session
from PBL5_NEW.extension import db
from PBL5_NEW.pbl5_ma import AccountSchema
from PBL5_NEW.model import Account

account_schema = AccountSchema()
accounts_schema = AccountSchema(many=True)

def add_account_service():
    data = request.json
    if data and 'username' in data and 'password' in data and 'role' in data:
        username = data['username']
        password = data['password']
        role = data['role']

        # Kiểm tra xem username đã tồn tại hay chưa
        existing_account = Account.query.filter_by(username=username).first()
        if existing_account:
            return jsonify({"message": "Username already exists"}), 400

        try:
            new_account = Account(username=username, password=password, role=role)
            db.session.add(new_account)
            db.session.commit()
            return jsonify({
                "message": "Account added successfully!",
                "id_account": new_account.id_account,
                "role": new_account.role
            }), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
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
        if data and 'password' in data and 'role' in data:
            try:
                account.password = data['password']
                account.role = data['role']
                db.session.commit()
                return jsonify({"message": "Account updated successfully!"}), 200
            except Exception as e:
                db.session.rollback()
                print(f"Error: {e}")
                return jsonify({"message": "Could not update account!", "error": str(e)}), 400
        else:
            return jsonify({"message": "Request error: Missing fields"}), 400
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
            print(f"Error: {e}")
            return jsonify({"message": "Could not delete account!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Account not found"}), 404

def check_account_service():
    data = request.json
    account = Account.query.filter_by(username=data['username']).first()
    if account is None:
        return jsonify({"message": "Account not found"}), 400
    else:
        if account.password != data['password']:
            return jsonify({"message": "Password is incorrect"}), 400
        else:
            session['id_account'] = account.id_account  # Lưu giá trị session
            return jsonify({
                "message": "Login successful",
                "role": account.role
            }), 200

def check_username_exists_service(username):
    existing_account = Account.query.filter_by(username=username).first()
    if existing_account:
        return jsonify({"exists": True}), 200
    else:
        return jsonify({"exists": False}), 200

def change_password_service():
    data = request.json
    if 'id_account' not in session:
        return jsonify({"message": "Unauthorized"}), 401

    id_account = session['id_account']
    old_password = data.get('old_password')
    new_password = data.get('new_password')

    account = Account.query.get(id_account)
    if not account:
        return jsonify({"message": "Account not found"}), 404

    if account.password != old_password:
        return jsonify({"message": "Old password is incorrect"}), 400

    try:
        account.password = new_password
        db.session.commit()
        return jsonify({"message": "Password changed successfully!"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Could not change password!", "error": str(e)}), 400
