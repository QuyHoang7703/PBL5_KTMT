from .extension import db

class Account(db.Model):
    id_account = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(100), nullable=False)

    def __init__(self, username, password, name, phone):
        self.username = username
        self.password = password
        self.name = name
        self.phone = phone

class Customer(db.Model):
    id_customer = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.Boolean, nullable=False)  # Assuming 0 for male, 1 for female
    phone = db.Column(db.String(10), nullable=False)
    cccd = db.Column(db.String(10), nullable=False)

    def __init__(self, name, gender, phone, cccd):
        self.name = name
        self.gender = gender
        self.phone = phone
        self.cccd = cccd

class History(db.Model):
    id_history = db.Column(db.Integer, primary_key=True)
    vehicle_plate = db.Column(db.String(20), nullable=False)
    date_in = db.Column(db.Date, nullable=True)
    date_out = db.Column(db.Date, nullable=True)
    time_in = db.Column(db.Time, nullable=True)
    time_out = db.Column(db.Time, nullable=True)

    def __init__(self, vehicle_plate, date_in, date_out, time_in, time_out):
        self.vehicle_plate = vehicle_plate
        self.date_in = date_in
        self.date_out = date_out
        self.time_in = time_in
        self.time_out = time_out

class Ticket(db.Model):
    vehicle_plate = db.Column(db.String(20), primary_key=True)
    init_date = db.Column(db.Date, nullable=False)
    expiry = db.Column(db.Date, nullable=False)
    id_customer = db.Column(db.Integer, db.ForeignKey('customer.id_customer'), nullable=False)
    customer = db.relationship('Customer', backref=db.backref('tickets', lazy=True))

    def __init__(self, vehicle_plate, init_date, expiry, id_customer):
        self.vehicle_plate = vehicle_plate
        self.init_date = init_date
        self.expiry = expiry
        self.id_customer = id_customer
