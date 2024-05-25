from flask import request, jsonify
from PBL5.extension import db
from PBL5.pbl5_ma import CustomerSchema
from PBL5.model import Customer
import json

customer_schema = CustomerSchema()
customers_schema = CustomerSchema(many=True)

def add_customer_service():
    data = request.json
    if data and 'name' in data and 'gender' in data and 'phone' in data and 'cccd' in data:
        name = data['name']
        gender = data['gender']
        phone = data['phone']
        cccd = data['cccd']
        try:
            new_customer = Customer(name, gender, phone, cccd)
            db.session.add(new_customer)
            db.session.commit()
            return jsonify({"message": "Customer added successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"message": "Could not add customer!"}), 400
    else:
        return jsonify({"message": "Request error"}), 400

def get_customer_by_id_service(id):
    customer = Customer.query.get(id)
    if customer:
        return customer_schema.jsonify(customer)
    else:
        return jsonify({"message": "Customer not found"}), 404

def get_all_customers_service():
    customers = Customer.query.all()
    return customers_schema.jsonify(customers)

def update_customer_by_id_service(id):
    customer = Customer.query.get(id)
    data = request.json
    if customer:
        if data and 'name' in data:
            try:
                customer.name = data['name']
                customer.gender = data['gender']
                customer.phone = data['phone']
                customer.cccd = data['cccd']
                db.session.commit()
                return jsonify({"message": "Customer updated successfully!"}), 200
            except Exception as e:
                db.session.rollback()
                return jsonify({"message": "Could not update customer!"}), 400
    else:
        return jsonify({"message": "Customer not found"}), 404

def delete_customer_by_id_service(id):
    customer = Customer.query.get(id)
    if customer:
        try:
            db.session.delete(customer)
            db.session.commit()
            return jsonify({"message": "Customer deleted successfully!"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"message": "Could not delete customer!"}), 400
    else:
        return jsonify({"message": "Customer not found"}), 404
