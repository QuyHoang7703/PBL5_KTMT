from flask import Blueprint
from .services import (
    add_ticket_service,
    get_ticket_by_vehicle_plate_service,
    get_all_tickets_service,
    update_ticket_by_vehicle_plate_service,
    delete_ticket_by_vehicle_plate_service,
    get_vehicle_plates_by_cccd_service,
    get_tickets_by_id_customer_service,
    add_ticket_by_admin_service
)

tickets = Blueprint("tickets", __name__)

# Add new ticket
@tickets.route("/ticket-management/ticket", methods=['POST'])
def add_ticket():
    return add_ticket_service()

@tickets.route("/ticket-management/ticket_admin", methods=['POST'])
def add_ticket_by_admin():
    return add_ticket_by_admin_service()


# Get ticket by vehicle plate
@tickets.route("/ticket-management/ticket/<string:vehicle_plate>", methods=['GET'])
def get_ticket_by_vehicle_plate(vehicle_plate):
    return get_ticket_by_vehicle_plate_service(vehicle_plate)

# Get all tickets
@tickets.route("/ticket-management/tickets", methods=['GET'])
def get_all_tickets():
    return get_all_tickets_service()

# Update ticket
@tickets.route("/ticket-management/ticket/<string:vehicle_plate>", methods=['PUT'])
def update_ticket_by_vehicle_plate(vehicle_plate):
    return update_ticket_by_vehicle_plate_service(vehicle_plate)

# Delete ticket
@tickets.route("/ticket-management/ticket/<string:vehicle_plate>", methods=['DELETE'])
def delete_ticket_by_vehicle_plate(vehicle_plate):
    return delete_ticket_by_vehicle_plate_service(vehicle_plate)

# Get all vehicle plates by cccd
@tickets.route("/ticket-management/tickets/vehicle-plates/<string:cccd>", methods=['GET'])
def get_vehicle_plates_by_cccd(cccd):
    return get_vehicle_plates_by_cccd_service(cccd)

# Get tickets by id_customer
@tickets.route("/ticket-management/tickets/customer/<int:id_customer>", methods=['GET'])
def get_tickets_by_id_customer(id_customer):
    return get_tickets_by_id_customer_service(id_customer)