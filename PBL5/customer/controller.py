from flask import Blueprint
from .services import (add_customer_service, get_customer_by_id_service,
                       get_all_customers_service, update_customer_by_id_service,
                       delete_customer_by_id_service)
customers = Blueprint("customers", __name__)

# Add a new customer
@customers.route("/customer-management/customer", methods=['POST'])
def add_customer():
    return add_customer_service()

# Get customer by id
@customers.route("/customer-management/customer/<int:id>", methods=['GET'])
def get_customer_by_id(id):
    return get_customer_by_id_service(id)

# Get all customers
@customers.route("/customer-management/customers", methods=['GET'])
def get_all_customers():
    return get_all_customers_service()

# Update customer
@customers.route("/customer-management/customer/<int:id>", methods=['PUT'])
def update_customer_by_id(id):
    return update_customer_by_id_service(id)

# Delete customer
@customers.route("/customer-management/customer/<int:id>", methods=['DELETE'])
def delete_customer_by_id(id):
    return delete_customer_by_id_service(id)
