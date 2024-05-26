import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from skimage.filters import threshold_local
import math
import time
from ultralytics import YOLO

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
        license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])
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

stream_url = "http://192.168.1.16:81/stream"
yolo_model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
character_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40_v13_ket_hop.keras")
def process_camera_stream(stream_url, model, my_model):
    while True:
        try:
            vid = initialize_camera_stream(stream_url)
            while True:
                ret, frame = vid.read()
                if not ret or frame is None:
                    print("Không thể nhận frame")
                    break
                results = model.predict(frame, show=False, stream=True, device='0')
                if results:
                    X, Y, W, H = None, None, None, None
                    for result in results:
                        boxes = result.boxes.cpu().numpy()
                        if len(boxes) == 0:
                            continue
                        for box in boxes:
                            X = box.xyxy[0][0]
                            Y = box.xyxy[0][1]
                            W = box.xywh[0][2]
                            H = box.xywh[0][3]
                    if X is not None and Y is not None and W is not None and H is not None:
                        img_crop = frame[int(Y) - 2: int(Y) + int(H) + 4, int(X) - 2: int(X) + int(W) + 4]
                        if img_crop is None or img_crop.size == 0:
                            print("Vùng cắt trống hoặc không hợp lệ")
                            continue
                        A, B = find_bottom_points(img_crop)
                        img_crop = rotate_image(img_crop, A, B)
                        V = cv2.split(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV))[2]
                        T = threshold_local(V, 35, offset=10, method="gaussian")
                        thresh = (V > T).astype("uint8") * 255
                        thresh = cv2.bitwise_not(thresh)
                        thresh = imutils.resize(thresh, width=390)
                        thresh = cv2.medianBlur(thresh, 5)
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
                        print("Num_labels:", num_labels)
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
                                if 0.2 < aspect_ratio < 1.0 and 1.0 > solidity > 0.3 and 0.2 < height_ratio < 1.0:
                                    character = np.array(mask[y - 2: y + h + 4, x - 2:x + w + 4])
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
                            predicted_character = classes[predicted_index]
                            predicted_characters.append(predicted_character)
                            formatted_candidates.append((predicted_character, coords))
                        bien_so_du_doan = format_license_plate(formatted_candidates)
                        print(bien_so_du_doan)
                        cv2.rectangle(frame, (int(X), int(Y)), (int(X + W), int(Y + H)), (0, 0, 255, 2))
                        cv2.putText(frame, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Segmentation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vid.release()
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")
            time.sleep(0.5)  # Đợi một giây trước khi thử lại

# Call the function
# process_camera_stream(stream_url, model, my_model)

# Cấu hình và chạy quá trình xử lý video

process_camera_stream(stream_url, yolo_model, character_model)
