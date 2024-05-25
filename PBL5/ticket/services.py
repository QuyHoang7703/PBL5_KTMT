from flask import request, jsonify
from PBL5.extension import db
from PBL5.pbl5_ma import TicketSchema
from PBL5.model import Ticket
import json
from datetime import datetime

ticket_schema = TicketSchema()
tickets_schema = TicketSchema(many=True)

def add_ticket_service():
    data = request.json
    if data and 'vehicle_plate' in data and 'init_date' in data and 'expiry' in data and 'id_customer' in data:
        vehicle_plate = data['vehicle_plate']
        init_date = datetime.strptime(data['init_date'], "%Y-%m-%d").date()
        expiry = datetime.strptime(data['expiry'], "%Y-%m-%d").date()
        id_customer = data['id_customer']
        try:
            new_ticket = Ticket(vehicle_plate, init_date, expiry, id_customer)
            db.session.add(new_ticket)
            db.session.commit()
            return jsonify({"message": "Ticket added successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"message": f"Could not add ticket! Error: {str(e)}"}), 400
    else:
        return jsonify({"message": "Request error"}), 400


def get_ticket_by_id_service(id):
    ticket = Ticket.query.get(id)
    if ticket:
        return ticket_schema.jsonify(ticket)
    else:
        return jsonify({"message": "Ticket not found"}), 404

def get_all_tickets_service():
    tickets = Ticket.query.all()
    return tickets_schema.jsonify(tickets)

def update_ticket_by_id_service(id):
    ticket = Ticket.query.get(id)
    data = request.json
    if ticket:
        if data and 'vehicle_plate' in data:
            try:
                ticket.vehicle_plate = data['vehicle_plate']
                ticket.init_date = datetime.strptime(data['init_date'], "%Y-%m-%d").date()
                ticket.expiry = datetime.strptime(data['expiry'], "%Y-%m-%d").date()
                ticket.id_customer = data['id_customer']
                db.session.commit()
                return jsonify({"message": "Ticket updated successfully!"}), 200
            except Exception as e:
                db.session.rollback()
                return jsonify({"message": f"Could not update ticket! Error: {str(e)}"}), 400
    else:
        return jsonify({"message": "Ticket not found"}), 404


def delete_ticket_by_id_service(id):
    ticket = Ticket.query.get(id)
    if ticket:
        try:
            db.session.delete(ticket)
            db.session.commit()
            return jsonify({"message": "Ticket deleted successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"message": "Could not delete ticket!"}), 400
    else:
        return jsonify({"message": "Ticket not found"}), 404
