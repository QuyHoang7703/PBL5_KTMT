from flask import Flask, Blueprint, render_template, Response , session , jsonify
from .extension import db, ma
from .model import Account, Ticket, History, Information
import os
from .account.controller import accounts
from .ticket.controller import tickets
from .information.controller import information
from .history.controller import histories
import requests,threading
from .stream_cam import generate_frames,stream_url_vao,yolo_model,character_model,stream_url_ra

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database_pbl5.db'
    app.config['SECRET_KEY'] = 'pbl5'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # Khởi tạo extension db với ứng dụng
    db.init_app(app)
    ma.init_app(app)
    # Định nghĩa hàm create_db trong phạm vi của ứng dụng
    def create_db():
        if not os.path.exists("instance/database_pbl5.db"):
            with app.app_context():
                db.create_all()
            print("database created successfully")

    # Gọi hàm create_db để tạo cơ sở dữ liệu
    create_db()
    app.register_blueprint(accounts)
    app.register_blueprint(information)
    app.register_blueprint(tickets)
    app.register_blueprint(histories)

    base_url = 'http://192.168.234.130:5000'

    @app.route('/')
    def home():
        return render_template('/login.html', base_url=base_url)

    @app.route('/index')
    def index():
        return render_template('/index.html', base_url=base_url)

    @app.route('/index_2')
    def index_2():
        return render_template('/index_2.html', base_url=base_url)

    @app.route('/customer')
    def customer():
        return render_template('/customer.html', base_url=base_url)

    @app.route('/ticket')
    def ticket():
        return render_template('/ticket.html', base_url=base_url)

    @app.route('/registervehicle')
    def registervehicle():
        return render_template('/RegisterVehicle.html', base_url=base_url)

    @app.route('/history')
    def history():
        return render_template('/history.html', base_url=base_url )

    @app.route('/history_2')
    def history_2():
        return render_template('/History_2.html', base_url=base_url)


    @app.route('/get_session', methods=['GET'])
    def get_session():
        if 'id_account' in session:
            return jsonify({"id_account": session['id_account']}), 200
        else:
            return jsonify({"message": "No session data found"}), 400

    @app.route('/information_detail')
    def information_detail():
        return render_template('/information.html', base_url=base_url)

    @app.route('/video_feed1')
    def video_feed1():
        return Response(generate_frames(stream_url_vao), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/video_feed2')
    def video_feed2():
        return Response(generate_frames(stream_url_ra), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/camera')
    def camera():
        return render_template('camera.html')

    thread_1 = threading.Thread(target=generate_frames, args=(stream_url_vao,))
    thread_2 = threading.Thread(target=generate_frames, args=(stream_url_ra,))
    thread_1.start()
    thread_2.start()

    # def start_camera_streams():
    #     # Khởi động các luồng camera với daemon=True để chạy nền
    #     threading.Thread(target=generate_frames, args=(stream_url_vao,), daemon=True).start()
    #     threading.Thread(target=generate_frames, args=(stream_url_ra,), daemon=True).start()
    #     print("Camera streams started in background.")
    #
    # start_camera_streams()

    return app
