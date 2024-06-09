import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from skimage.filters import threshold_local
import math
import time
import requests
from ultralytics import YOLO
from datetime import datetime

def format_license_plate(candidates):
    first_line = []
    second_line = []
    candidates = sorted(candidates, key=lambda x: x[1][1])
    threshold_y = candidates[0][1][1] + 40
    for candidate, (x, y) in candidates:
        if y < threshold_y:
            first_line.append((candidate, x))
        else:
            second_line.append((candidate, x))
    first_line = sorted(first_line, key=lambda x: x[1])
    second_line = sorted(second_line, key=lambda x: x[1])
    if len(second_line) == 0:
        license_plate = "".join([str(ele[0]) for ele in first_line])
    else:
        license_plate = "".join([str(ele[0]) for ele in first_line])  + "".join(
            [str(ele[0]) for ele in second_line])
    return license_plate


def find_bottom_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        return "Không tìm thấy contour"
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    sorted_box_y = sorted(box, key=lambda x: x[1])
    bottom_points = sorted(sorted_box_y[-2:], key=lambda x: x[0])
    return bottom_points[0], bottom_points[1]


def rotate_image(img, bottom_left, bottom_right):
    dx = abs(bottom_right[0] - bottom_left[0])
    dy = abs(bottom_right[1] - bottom_left[1])
    angle = math.atan(dy / dx) * (180.0 / math.pi)
    if bottom_left[1] < bottom_right[1]:
        angle = np.arctan2(dy, dx) * 180 / np.pi
    else:
        angle = -np.arctan2(dy, dx) * 180 / np.pi
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return rotated_img


def initialize_camera_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise Exception("Không thể mở luồng video")
    return cap

last_detection_time = 0
detection_result = None
confidence = 0
detection_confirmed = False
confirmation_time = 2
confidence_threshold = 0.8
previous_license_plate = None

stream_url = "http://192.168.174.106:81/stream"
yolo_model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
character_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40_v13_ket_hop.keras")


def generate_frames(stream_url, model, my_model):
    global last_detection_time, detection_result, detection_confirmed, previous_license_plate
    vid = initialize_camera_stream(stream_url)
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        results = model.predict(frame, show=False, stream=True, device='0')
        if results:
            X, Y, W, H = None, None, None, None
            for result in results:
                boxes = result.boxes.cpu().numpy()
                if len(boxes) == 0:
                    continue
                for box in boxes:
                    if box.conf>0.9 and len(box.xyxy>=2) and len(box.xywh>=2):
                        X = box.xyxy[0][0]
                        Y = box.xyxy[0][1]
                        W = box.xywh[0][2]
                        H = box.xywh[0][3]
            if X is not None and Y is not None and W is not None and H is not None:
                img_crop = frame[int(Y) - 2: int(Y) + int(H) + 4, int(X) - 2: int(X) + int(W) + 4]
                if img_crop is None or img_crop.size == 0:
                    continue
                V = cv2.split(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV))[2]
                T = threshold_local(V, 35, offset=12, method="gaussian")
                thresh = (V > T).astype("uint8") * 255
                thresh = cv2.bitwise_not(thresh)
                thresh = imutils.resize(thresh, width=390)
                thresh = cv2.medianBlur(thresh, 5)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
                candidates = []
                for label in range(1, num_labels):
                    mask = np.zeros(thresh.shape, dtype=np.uint8)
                    mask[labels == label] = 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) > 0:
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / float(h)
                            solidity = cv2.contourArea(contour) / float(w * h)
                            height_ratio = h / float(thresh.shape[0])
                            if 0.2 < aspect_ratio < 1.0 and 1.0 > solidity > 0.3 and 0.2 < height_ratio < 1.0:
                                character = np.array(mask[y - 2: y + h + 4, x - 2:x + w + 4])
                                if character.size > 0:
                                    character_resized = cv2.resize(character, (30, 40), interpolation=cv2.INTER_AREA)
                                    character_normalized = character_resized / 255.0
                                    character_input = np.expand_dims(character_normalized, axis=-1)
                                    candidates.append((character_input, (x, y)))
                predicted_characters = []
                formatted_candidates = []
                if candidates:
                    for character_input, coords in candidates:
                        character_input = np.expand_dims(character_input, axis=0)
                        prediction = my_model.predict(character_input)
                        predicted_index = np.argmax(prediction)
                        predicted_character = classes[predicted_index]
                        predicted_characters.append(predicted_character)
                        formatted_candidates.append((predicted_character, coords))
                    bien_so_du_doan = format_license_plate(formatted_candidates)

                    current_time = time.time()
                    if bien_so_du_doan == previous_license_plate:
                        if (current_time - last_detection_time) >= confirmation_time:
                            detection_confirmed = True
                    else:
                        previous_license_plate = bien_so_du_doan
                        detection_result = bien_so_du_doan
                        last_detection_time = current_time
                        detection_confirmed = False

                    if detection_confirmed:
                        url = 'http://192.168.174.130:5000/history-management/history'
                        current_time = datetime.now()
                        data = {
                            'vehicle_plate': detection_result,
                            'date': current_time.strftime("%d-%m-%Y"),
                            'time': current_time.strftime("%H:%M:%S")
                        }
                        response = requests.post(url, json=data)
                        if response.status_code == 200:
                            print("Dữ liệu đã được gửi thành công!")
                        else:
                            print("Đã xảy ra lỗi:", response.status_code)
                        detection_confirmed = False

                    cv2.rectangle(frame, (int(X), int(Y)), (int(X + W), int(Y + H)), (0, 0, 255, 2))
                    cv2.putText(frame, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
