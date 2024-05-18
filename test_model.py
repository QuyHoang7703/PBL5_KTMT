import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from ultralytics import YOLO
import imutils
import datetime
from skimage.filters import threshold_local
import time
import math
import torch
import requests

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")


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
        license_plate = "".join([str(ele[0]) for ele in first_line]) + "".join([str(ele[0]) for ele in second_line])
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


def send_to_server(data):878
    try:
        response = requests.post("http://localhost:5000/api/receive", json=data)  # URL của server Flask
        response.raise_for_status()  # Kiểm tra mã trạng thái HTTP
        return response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data: {e}")
        return None


last_detection_time = 0
detection_result = None
confidence = 0
detection_confirmed = False
confirmation_time = 5
confidence_threshold = 0.8
previous_license_plate = None


def process_frame(frame, model, my_model, classes):
    global last_detection_time, detection_result, detection_confirmed, previous_license_plate
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
        if X is not None and Y is not None and W is not None and H is not None:
            img_crop = frame[int(Y) - 2: int(Y) + int(H) + 4, int(X) - 2: int(X) + int(W) + 4]
            A, B = find_bottom_points(img_crop)
            # if A == "Không tìm thấy contour" or B == "Không tìm thấy contour":
            #     return frame
            img_crop = rotate_image(img_crop, A, B)
            V = cv2.split(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV))[2]
            T = threshold_local(V, 35, offset=14, method="gaussian")
            thresh = (V > T).astype("uint8") * 255
            thresh = cv2.bitwise_not(thresh)
            thresh = imutils.resize(thresh, width=400)
            thresh = cv2.medianBlur(thresh, 5)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            candidates = []

            for label in range(1, num_labels):
                mask = np.zeros(thresh.shape, dtype=np.uint8)
                mask[labels == label] = 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    solidity = cv2.contourArea(contour) / float(w * h)
                    height_ratio = h / float(thresh.shape[0])
                    if 0.21 < aspect_ratio < 1.0 and 1.0 > solidity >= 0.3 and 0.2 < height_ratio < 1.0:
                        character = np.array(mask[y - 2: y + h + 4, x - 2: x + w + 4])
                        if character.size > 0:
                            character_resized = cv2.resize(character, (30, 40), interpolation=cv2.INTER_AREA)
                            character_normalized = character_resized / 255.0
                            character_input = np.expand_dims(character_normalized, axis=-1)
                            candidates.append((character_input, (x, y)))

            predicted_characters = []
            formatted_candidates = []
            for character_input, coords in candidates:
                character_input = np.expand_dims(character_input, axis=0)
                prediction = my_model.predict(character_input)
                predicted_index = np.argmax(prediction)
                max_probability = np.max(prediction)# Them 2 dong
                print("Ti le : ", max_probability)
                if max_probability >= 0.9:
                    predicted_character = classes[predicted_index]
                    predicted_characters.append(predicted_character)
                    formatted_candidates.append((predicted_character, coords))

            bien_so_du_doan = format_license_plate(formatted_candidates)
            print(bien_so_du_doan)

            current_time = time.time()
            if bien_so_du_doan == previous_license_plate:
                if (current_time - last_detection_time) >= confirmation_time:
                    detection_confirmed = True
            else:
                previous_license_plate = bien_so_du_doan  # Lưu trữ biển số hiện tại
                detection_result = bien_so_du_doan
                last_detection_time = current_time
                detection_confirmed = False

            if detection_confirmed:
                data_to_send = {
                    "license_plate": detection_result,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                status = send_to_server(data_to_send)
                if status == 200:
                    print("Data sent successfully.")
                else:
                    print("Failed to send data.")
                detection_confirmed = False

            detection_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("Thời gian nhận diện:", detection_time)

            cv2.rectangle(frame, (int(X), int(Y)), (int(X + W), int(Y + H)), (0, 0, 255, 2))
            cv2.putText(frame, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            time.sleep(0)
    return frame


model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40_v13_ket_hop.keras")

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ camera.")
            break
        frame = process_frame(frame, model, my_model, classes)
        cv2.imshow("Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

cap.release()
cv2.destroyAllWindows()
