from flask import request, jsonify
from PBL5_NEW.extension import db
from PBL5_NEW.pbl5_ma import TicketSchema
from PBL5_NEW.model import Ticket, Information
from datetime import datetime

ticket_schema = TicketSchema()
tickets_schema = TicketSchema(many=True)

def add_ticket_service():
    data = request.json
    if data and 'vehicle_plate' in data and 'init_date' in data and 'expiry' in data and 'status' in data and 'id_customer' in data:
        vehicle_plate = data['vehicle_plate']
        status = data['status']
        id_customer = data['id_customer']
        init_date = datetime.strptime(data['init_date'], "%Y-%m-%d").date()
        expiry = datetime.strptime(data['expiry'], "%Y-%m-%d").date()
        # Kiểm tra xem cccd có tồn tại hay không
        existing_information = Information.query.filter_by(id_account=id_customer).first()
        if not existing_information:
            return jsonify({"message": "Customer is not found"}), 400

        # Kiểm tra xem vehicle_plate đã tồn tại hay chưa
        existing_ticket = Ticket.query.filter_by(vehicle_plate=vehicle_plate).first()
        if existing_ticket:
            return jsonify({"message": "Vehicle plate already exists"}), 400

        try:
            new_ticket = Ticket(vehicle_plate=vehicle_plate, init_date=init_date, expiry=expiry, status=status, cccd=existing_information.cccd)
            db.session.add(new_ticket)
            db.session.commit()
            return jsonify({"message": "Ticket added successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return jsonify({"message": "Could not add ticket!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Request error: Missing fields"}), 400

def get_ticket_by_vehicle_plate_service(vehicle_plate):
    ticket = Ticket.query.get(vehicle_plate)
    if ticket:
        return ticket_schema.jsonify(ticket)
    else:
        return jsonify({"message": "Ticket not found"}), 404

def get_all_tickets_service():
    tickets = Ticket.query.all()
    return tickets_schema.jsonify(tickets)

def update_ticket_by_vehicle_plate_service(vehicle_plate):
    ticket = Ticket.query.get(vehicle_plate)
    data = request.json
    if ticket:
        if data and 'init_date' in data and 'expiry' in data and 'status' in data:
            try:
                ticket.status = data['status']
                ticket.init_date = datetime.strptime(data['init_date'], "%Y-%m-%d").date()
                ticket.expiry = datetime.strptime(data['expiry'], "%Y-%m-%d").date()
                db.session.commit()
                return jsonify({"message": "Ticket updated successfully!"}), 200
            except Exception as e:
                db.session.rollback()
                print(f"Error: {e}")
                return jsonify({"message": "Could not update ticket!", "error": str(e)}), 400
        else:
            return jsonify({"message": "Request error: Missing fields"}), 400
    else:
        return jsonify({"message": "Ticket not found"}), 404

def delete_ticket_by_vehicle_plate_service(vehicle_plate):
    ticket = Ticket.query.get(vehicle_plate)
    if ticket:
        try:
            db.session.delete(ticket)
            db.session.commit()
            return jsonify({"message": "Ticket deleted successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return jsonify({"message": "Could not delete ticket!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Ticket not found"}), 404

def get_vehicle_plates_by_cccd_service(cccd):
    tickets = Ticket.query.filter_by(cccd=cccd).all()
    if tickets:
        vehicle_plates = [ticket.vehicle_plate for ticket in tickets]
        return jsonify({"vehicle_plates": vehicle_plates}), 200
    else:
        return jsonify({"message": "No tickets found for the provided cccd"}), 404

def get_tickets_by_id_customer_service(id_customer):
    # Lấy thông tin khách hàng dựa trên id_customer
    existing_information = Information.query.filter_by(id_account=id_customer).first()
    if not existing_information:
        return jsonify({"message": "Customer not found"}), 404

    # Lấy cccd từ thông tin khách hàng
    cccd = existing_information.cccd

    # Truy vấn tất cả các ticket liên quan đến cccd
    tickets = Ticket.query.filter_by(cccd=cccd).all()
    if tickets:
        return tickets_schema.jsonify(tickets), 200
    else:
        return jsonify({"message": "No tickets found for the provided customer ID"}), 404

def add_ticket_by_admin_service():
    data = request.json
    if data and 'vehicle_plate' in data and 'init_date' in data and 'expiry' in data and 'status' in data and 'cccd' in data:
        vehicle_plate = data['vehicle_plate']
        status = data['status']
        cccd = data['cccd']
        init_date = datetime.strptime(data['init_date'], "%Y-%m-%d").date()
        expiry = datetime.strptime(data['expiry'], "%Y-%m-%d").date()

        existing_information = Information.query.filter_by(cccd=cccd).first()
        if not existing_information:
            return jsonify({"message": "Customer not found"}), 400

        existing_ticket = Ticket.query.filter_by(vehicle_plate=vehicle_plate).first()
        if existing_ticket:
            return jsonify({"message": "Vehicle plate already exists"}), 400

        try:
            new_ticket = Ticket(vehicle_plate=vehicle_plate, init_date=init_date, expiry=expiry, status=status, cccd=cccd)
            db.session.add(new_ticket)
            db.session.commit()
            return jsonify({"message": "Ticket added successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return jsonify({"message": "Could not add ticket!", "error": str(e)}), 400
    else:
        return jsonify({"message": "Request error: Missing fields"}), 400