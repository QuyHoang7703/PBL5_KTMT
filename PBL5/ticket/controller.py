from flask import Blueprint
from .services import (add_ticket_service, get_ticket_by_id_service,
                       get_all_tickets_service, update_ticket_by_id_service,
                       delete_ticket_by_id_service)
tickets = Blueprint("tickets", __name__)

# Add a new ticket
@tickets.route("/ticket-management/ticket", methods=['POST'])
def add_ticket():
    return add_ticket_service()

# Get ticket by id
@tickets.route("/ticket-management/ticket/<string:id>", methods=['GET'])
def get_ticket_by_id(id):
    return get_ticket_by_id_service(id)

# Get all tickets
@tickets.route("/ticket-management/tickets", methods=['GET'])
def get_all_tickets():
    return get_all_tickets_service()

# Update ticket
@tickets.route("/ticket-management/ticket/<string:id>", methods=['PUT'])
def update_ticket_by_id(id):
    return update_ticket_by_id_service(id)

# Delete ticket
@tickets.route("/ticket-management/ticket/<string:id>", methods=['DELETE'])
def delete_ticket_by_id(id):
    return delete_ticket_by_id_service(id)
