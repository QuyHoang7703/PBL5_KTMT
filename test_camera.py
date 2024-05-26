
import urllib.request

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from ultralytics import YOLO
import imutils
from skimage.filters import threshold_local
import time
def correct_tilt(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh có thể giúp cải thiện việc phát hiện cạnh
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 30:  # Giới hạn góc xoay để tránh những thay đổi quá lớn
                angles.append(angle)

        if len(angles) > 0:
            median_angle = np.median(angles)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            corrected_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return corrected_img
    return image
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
url = 'http://10.10.59.136/stream'
model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40_v13_ket_hop.keras")
X, Y, W, H = None, None, None, None
img_crop=None
while True:
    try:
        img_req = urllib.request.urlopen(url)
        img_np =np.array(bytearray(img_req.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        # print(img_frame)
        # cv2.imshow("Nhan dien bien so xe", img_frame)
        frame = cv2.resize(frame, (800, 600))
        results = model.predict(frame, show=True, stream=True, device='0')
        if results:

            X, Y, W, H = None, None, None, None
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    X = box.xyxy[0][0]
                    Y = box.xyxy[0][1]
                    W = box.xywh[0][2]
                    H = box.xywh[0][3]
            if X != None and Y != None and W != None and H != None:
                img_crop = frame[int(Y) - 2: int(Y) + int(H) + 4, int(X) - 2: int(X) + int(W) + 4]
                # img_crop = correct_tilt(img_crop)
                V = cv2.split(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV))[2]
                T = threshold_local(V, 35, offset=10, method="gaussian")
                thresh = (V > T).astype("uint8") * 255
                thresh = cv2.bitwise_not(thresh)
                thresh = imutils.resize(thresh, width=390)
                thresh = cv2.medianBlur(thresh, 5)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
                print("Num_labels:", num_labels)
                # Khởi tạo danh sách để lưu các ký tự ứng viên và bounding rectangles
                candidates = []
                bounding_rects = []

                # Lặp qua các nhãn từ 1 đến num_labels - 1 (loại bỏ nhãn của background)
                for label in range(1, num_labels):
                    # Tạo mask chứa các pixel có nhãn cùng là label
                    mask = np.zeros(thresh.shape, dtype=np.uint8)
                    mask[labels == label] = 255  # Các các pixel cùng nhãn giá trị 255

                    # Tìm contours từ mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)

                        aspect_ratio = w / float(h)
                        solidity = cv2.contourArea(contour) / float(w * h)
                        height_ratio = h / float(thresh.shape[0])

                        # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
                        if 0.2 < aspect_ratio < 1.0 and 1.0 > solidity > 0.3 and 0.2 < height_ratio < 1.0:
                            # bounding_rects.append((x, y, w, h))
                            bounding_rects.append((x - 1, y - 1, w + 2, h + 2))

                            # Trích xuất ký tự
                            # character = binary[y-3: y + h+3, x-3:x + w+3]
                            character = np.array(mask[y - 2: y + h + 4, x - 2:x + w + 4])
                            # character = binary[y : y + h , x :x + w ]
                            # Đảm bảo kích thước ảnh phù hợp với mô hình
                            if character.size > 0:
                                character_resized = cv2.resize(character, (30, 40), interpolation=cv2.INTER_AREA)
                                # Chuẩn hóa giá trị pixel về khoảng [0, 1]
                                character_normalized = character_resized / 255.0
                                # Mở rộng chiều dữ liệu để phù hợp với input_shape của mô hình (32, 32, 1)
                                character_input = np.expand_dims(character_normalized, axis=-1)
                                # Thêm ký tự đã chuẩn bị vào danh sách các ký tự
                                # candidates.append(character_input)
                                candidates.append((character_input, (x, y)))


                # Dự đoán các ký tự từ danh sách các ký tự ứng viên
                predicted_characters = []
                formatted_candidates = []
                for character_input, coords in candidates:
                    character_input = np.expand_dims(character_input, axis=0)
                    prediction = my_model.predict(character_input)
                    # prediction = my_model.predict(np.array(character_input))  # Đảm bảo dữ liệu đầu vào đúng cho mô hình
                    predicted_index = np.argmax(prediction)
                    predicted_character = classes[predicted_index]
                    predicted_characters.append(predicted_character)
                    formatted_candidates.append((predicted_character, coords))  # Lưu ký tự và tọa độ

                # Sắp xếp và định dạng biển số xe
                bien_so_du_doan = format_license_plate(formatted_candidates)
                print(bien_so_du_doan)

                cv2.rectangle(frame, (int(X), int(Y)), (int(X + W), int(Y + H)), (0, 0, 255, 2))
                cv2.putText(frame, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                # time.sleep(0.5)
            cv2.imshow("Segmentation", frame)


    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cap.release()
    #     cv2.destroyAllWindows()
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         frame.release()
cv2.destroyAllWindows()
    #         break