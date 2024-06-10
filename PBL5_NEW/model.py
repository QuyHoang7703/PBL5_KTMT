from .extension import db

class Account(db.Model):
    __tablename__ = 'account'
    
    id_account = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.Boolean, nullable=False)  # False for customer, True for admin

    def __init__(self, username, password, role):
        self.username = username
        self.password = password
        self.role = role


class Information(db.Model):
    __tablename__ = 'information'
    
    id_account = db.Column(db.Integer, db.ForeignKey('account.id_account'), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.Boolean, nullable=False)  # False for male, True for female
    phone = db.Column(db.String(10), nullable=False)
    cccd = db.Column(db.String(10), unique=True, nullable=False)  # Identity card
    gmail = db.Column(db.String(100), nullable=False)

    account = db.relationship('Account', backref=db.backref('information', lazy=True))

    def __init__(self, id_account, name, gender, phone, cccd, gmail):
        self.id_account = id_account
        self.name = name
        self.gender = gender
        self.phone = phone
        self.cccd = cccd
        self.gmail = gmail


class Ticket(db.Model):
    __tablename__ = 'ticket'
    
    vehicle_plate = db.Column(db.String(20), primary_key=True, nullable=False)
    init_date = db.Column(db.Date, nullable=False)
    expiry = db.Column(db.Date, nullable=False)
    status = db.Column(db.Integer, nullable=False)  # False for pending, True for approved
    cccd = db.Column(db.String(10), db.ForeignKey('information.cccd'), nullable=False)

    information = db.relationship('Information', backref=db.backref('tickets', lazy=True))

    def __init__(self, vehicle_plate, init_date, expiry, status, cccd):
        self.vehicle_plate = vehicle_plate
        self.init_date = init_date
        self.expiry = expiry
        self.status = status
        self.cccd = cccd


class History(db.Model):
    __tablename__ = 'history'
    
    id_history = db.Column(db.Integer, primary_key=True, autoincrement=True)
    vehicle_plate = db.Column(db.String(20), db.ForeignKey('ticket.vehicle_plate'), nullable=False)
    date_in = db.Column(db.Date, nullable=True)
    date_out = db.Column(db.Date, nullable=True)
    time_in = db.Column(db.Time, nullable=True)
    time_out = db.Column(db.Time, nullable=True)

    ticket = db.relationship('Ticket', backref=db.backref('history', lazy=True))

    def __init__(self, vehicle_plate, date_in, date_out, time_in, time_out):
        self.vehicle_plate = vehicle_plate
        self.date_in = date_in
        self.date_out = date_out
        self.time_in = time_in
        self.time_out = time_out
