import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import threading

# Hàm để định dạng biển số xe từ danh sách các ứng viên
def format_license_plate(candidates):
    first_line = []
    second_line = []
    candidates = sorted(candidates, key=lambda x: x[1][1])  # Sắp xếp theo tọa độ y
    threshold_y = candidates[0][1][1] + 40  # Ngưỡng cho dòng đầu tiên

    for candidate, (x, y) in candidates:
        if y < threshold_y:
            first_line.append((candidate, x))
        else:
            second_line.append((candidate, x))

    first_line = sorted(first_line, key=lambda x: x[1])  # Sắp xếp lại theo tọa độ x
    second_line = sorted(second_line, key=lambda x: x[1])

    if len(second_line) == 0:
        license_plate = "".join([str(ele[0]) for ele in first_line])
    else:
        license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])
    return license_plate

# Hàm xử lý video capture trong một luồng riêng biệt
def process_video_capture():
    global vid, model, classes, my_model
    while True:
        try:
            ret, frame = vid.read()
            if not ret or frame is None:
                print("Không thể nhận frame")
                continue

            frame = cv2.resize(frame, (800, 600))
            results = model.predict(frame, show=False, stream=True, device='0')

            if results:
                X, Y, W, H = None, None, None, None
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        X = box.xyxy[0][0]
                        Y = box.xyxy[0][1]
                        W = box.xywh[0][2]
                        H = box.xywh[0][3]
                print("alo")

            cv2.imshow('frame', frame)

        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Khởi tạo đối tượng video capture từ URL
vid = cv2.VideoCapture("http://10.10.59.136:81/stream")
# vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Đặt buffer size của video capture

# Load pre-trained YOLO model
model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")

# Load pre-trained license plate recognition model
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40_v13_ket_hop.keras")

# Danh sách các lớp
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]

# Khởi tạo và bắt đầu luồng xử lý video capture
video_thread = threading.Thread(target=process_video_capture)
video_thread.start()

# Chờ luồng xử lý video capture kết thúc
video_thread.join()

# Giải phóng video capture và đóng cửa sổ OpenCV
vid.release()
cv2.destroyAllWindows()
