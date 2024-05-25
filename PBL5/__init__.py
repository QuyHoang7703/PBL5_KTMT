from flask import Flask, Blueprint, render_template, Response
from .extension import db, ma
from .model import Account, Ticket, History, Customer
import os
from .account.controller import accounts
from .ticket.controller import tickets
from .customer.controller import customers
from .history.controller import histories
from .webcam import Webcam
import requests

webcam = Webcam()


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_pbl5.db'
    app.config['SECRET_KEY'] = 'pbl5'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # Khởi tạo extension db với ứng dụng
    db.init_app(app)
    ma.init_app(app)
    # Định nghĩa hàm create_db trong phạm vi của ứng dụng
    def create_db():
        if not os.path.exists("instance/db_pbl5.db"):
            with app.app_context():
                db.create_all()
            print("db created successfully")
    
    # Gọi hàm create_db để tạo cơ sở dữ liệu
    create_db()
    app.register_blueprint(accounts)
    app.register_blueprint(tickets)
    app.register_blueprint(customers)
    app.register_blueprint(histories)
    
    @app.route('/')
    def home():
        return render_template('/login.html')  
    
    @app.route('/index')
    def index():
        return render_template('/index.html')  
    
    @app.route('/customer')
    def customer():
        return render_template('/customer.html')  
    
    @app.route('/ticket')
    def ticket():
        return render_template('/ticket.html')  
    
    @app.route('/history')
    def history():
        return render_template('/history.html')  
    
    @app.route('/camera')
    def camera():
        return render_template('/camera.html')  
    
    # def read_from_webcam():
    #     while True:
    #         image = next(webcam.get_frame())
    #         yield b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n--frame\r\n'
            
    # @app.route('/image_feed')
    # def image_feed():
    #     return Response(read_from_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")
    ESP32_URL = 'http://10.10.58.64/'

    def get_frame_from_esp32():
        # Get a frame from ESP32 camera
        response = requests.get(ESP32_URL)
        return response.content

    @app.route('/image_feed')
    def image_feed():
        return Response(get_frame_from_esp32(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return app
