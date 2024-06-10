from flask import request, jsonify
from PBL5_NEW.extension import db
from PBL5_NEW.pbl5_ma import InformationSchema
from PBL5_NEW.model import Information , Account

information_schema = InformationSchema()
informations_schema = InformationSchema(many=True)

def add_information_service():
    data = request.json
    if data and 'id_account' in data and 'name' in data and 'gender' in data and 'phone' in data and 'cccd' in data and 'gmail' in data:
        id_account = data['id_account']
        name = data['name']
        gender = data['gender']
        phone = data['phone']
        cccd = data['cccd']
        gmail = data['gmail']

        # Kiểm tra xem CCCD đã tồn tại hay chưa
        existing_information = Information.query.filter_by(cccd=cccd).first()
        if existing_information:
            return jsonify({"message": "CCCD already exists"}), 400

        try:
            new_information = Information(id_account=id_account, name=name, gender=gender, phone=phone, cccd=cccd, gmail=gmail)
            db.session.add(new_information)
            db.session.commit()
            return jsonify({"message": "Information added successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return jsonify({"message": "Could not add information!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Request error: Missing fields"}), 400

def get_information_by_cccd_service(cccd):
    information = Information.query.get(cccd)
    if information:
        return information_schema.jsonify(information)
    else:
        return jsonify({"message": "Information not found"}), 404

def get_all_informations_service():
    informations = Information.query.join(Account).filter(Account.role == 0).all()
    return informations_schema.jsonify(informations)

def update_information_by_cccd_service(cccd):
    information = Information.query.get(cccd)
    data = request.json
    if information:
        if data and 'name' in data and 'gender' in data and 'phone' in data and 'gmail' in data:
            try:
                information.name = data['name']
                information.gender = data['gender']
                information.phone = data['phone']
                information.gmail = data['gmail']
                db.session.commit()
                return jsonify({"message": "Information updated successfully!"}), 200
            except Exception as e:
                db.session.rollback()
                print(f"Error: {e}")
                return jsonify({"message": "Could not update information!", "error": str(e)}), 400
        else:
            return jsonify({"message": "Request error: Missing fields"}), 400
    else:
        return jsonify({"message": "Information not found"}), 404

def delete_information_by_cccd_service(cccd):
    information = Information.query.get(cccd)
    if information:
        try:
            db.session.delete(information)
            db.session.commit()
            return jsonify({"message": "Information deleted successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return jsonify({"message": "Could not delete information!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Information not found"}), 404

def get_information_by_id_account_service(id_account):
    information = Information.query.filter_by(id_account=id_account).first()
    if information:
        return information_schema.jsonify(information)
    else:
        return jsonify({"message": "Information not found"}), 404

def delete_information_by_id_account_service(id_account):
    information = Information.query.filter_by(id_account=id_account).first()
    if information:
        try:
            db.session.delete(information)
            db.session.commit()
            return jsonify({"message": "Information deleted successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return jsonify({"message": "Could not delete information!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Information not found"}), 404

def update_information_service(id_account):
    information = Information.query.filter_by(id_account=id_account).first()
    if not information:
        return jsonify({'message': 'Information not found'}), 404

    data = request.get_json()

    # Check if CCCD is changing and if it already exists elsewhere
    new_cccd = data.get('cccd')
    if new_cccd and new_cccd != information.cccd:
        existing_info = Information.query.filter(Information.cccd == new_cccd, Information.id_account != id_account).first()
        if existing_info:
            return jsonify({'message': 'CCCD already exists'}), 400

    try:
        information.name = data.get('name', information.name)
        information.gender = data.get('gender', information.gender)
        information.phone = data.get('phone', information.phone)
        if new_cccd:
            information.cccd = new_cccd
        information.gmail = data.get('gmail', information.gmail)
        db.session.commit()
        return jsonify({'message': 'Information updated successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Failed to update information', 'error': str(e)}), 500

