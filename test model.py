import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.filters import threshold_local
from ultralytics import YOLO
import imutils
import torch
import math
from imutils import perspective
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

path = r"D:\HocKy2-23_24\PBL5\Test Model"
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40_v13_ket_hop.keras")
model_yolo = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
def correct_tilt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh có thể giúp cải thiện việc phát hiện cạnh
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
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
            if abs(angle) < 60:  # Giới hạn góc xoay để tránh những thay đổi quá lớn
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
    threshold_y = candidates[0][1][1] + 50  # Ngưỡng cho dòng đầu tiên

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
def find_bottom_points(img):
    # Chuyển đổi ảnh sang ảnh grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để làm giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện các cạnh trong ảnh
    edges = cv2.Canny(blurred, 30, 150)

    # Tìm các contour trong ảnh
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:

        return "Không tìm thấy contour"
    # Tìm contour lớn nhất
    max_contour = max(contours, key=cv2.contourArea)

    # Xác định các đỉnh của contour lớn nhất
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Sắp xếp các đỉnh theo tọa độ y
    sorted_box_y = sorted(box, key=lambda x: x[1])

    # Lấy ra hai điểm phía dưới của hình chữ nhật
    bottom_points = sorted(sorted_box_y[-2:], key=lambda x: x[0])  # Sắp xếp theo tọa độ x

    return bottom_points[0], bottom_points[1]


def rotate_image(img, bottom_left, bottom_right):
    # Tính toán góc xoay dựa trên hai điểm dưới của hình chữ nhật
    dx = abs(bottom_right[0] - bottom_left[0])
    dy = abs(bottom_right[1] - bottom_left[1])
    # doi = abs(y1 - y2)
    # ke = abs(x1 - x2)
    angle = math.atan(dy / dx) * (180.0 / math.pi)


    if bottom_left[1] < bottom_right[1]:  # bottom_left cao hơn bottom_right
        angle = np.arctan2(dy, dx) * 180 / np.pi  # Góc dương (xoay theo chiều kim đồng hồ)
    else:
        angle = -np.arctan2(dy, dx) * 180 / np.pi  # Góc âm (xoay ngược chiều kim đồng hồ)

    # Tạo ma trận quay
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Áp dụng ma trận quay để xoay ảnh
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return rotated_img
def recognition_license_plate(img, img_path, my_model, model_yolo):
    results = model_yolo.predict(img)
    # if results is None or len(results.boxes.cpu().numpy()) == 0:
    #     print("No license plate detected.")
    #     return
    X, Y, W, H = None, None, None, None
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            X = box.xyxy[0][0]
            Y = box.xyxy[0][1]
            W = box.xywh[0][2]
            H = box.xywh[0][3]
    if X is not None and Y is not None and W is not None and H is not None:
        # img_crop = img[int(Y): int(Y) + int(H), int(X): int(X) + int(W)]
        # img_crop = img[int(Y) - 5: int(Y) + int(H) + 10, int(X) - 5: int(X) + int(W) + 10]
        img_crop = img[int(Y) - 2: int(Y) + int(H) + 4, int(X) - 2: int(X) + int(W) + 4]
        A, B = find_bottom_points(img_crop)
        img_crop = rotate_image(img_crop, A, B)
        # pts = [(X, Y), (X+W, Y), (X+W, Y+H), (X, Y+H)]
        # pts = np.array(pts)
        # img_crop = perspective.four_point_transform(img, pts)
        # Các bước xử lý tiếp theo...
    else:
        print("No license plate detected.")
        return
    # img_crop = img[int(Y) - 15: int(Y) + int(H) + 15, int(X) - 15: int(X) + int(W) + 15]
    # img_crop = correct_tilt(img_crop)
    # img_crop = process_image(img_crop)
    V = cv2.split(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV))[2]
    # print('Width: ', W)
    # print('Height: ', H)
    # t = W/H
    # print("Ngưỡng: ", t)
    # if t <= 0.5:
    #     block_size = int(W / 8) if int(W / 8) % 2 != 0 else int(W / 8) + 3
    #
    # else:
    #     block_size = int(W / 4) if int(W / 4) % 2 != 0 else int(W / 4) + 3
    #
    # total = W + H
    # my_offset = int(total / 10)
    # print("BLOCK SIZE ", block_size)
    # print("Offset ", my_offset)
    T = threshold_local(V, 35, offset=13, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=390)
    thresh = cv2.medianBlur(thresh, 5)

    candidates = []
    bounding_rects = []
    stat = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    for label in np.unique(labels):
        # if this is background label, ignore it
        if label == 0:
            continue

        # init mask to store the location of the character candidates
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

        # Tìm contours từ mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            height_ratio = h / float(thresh.shape[0])

            # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
            if 0.21 < aspect_ratio < 1.0 and 1.0 > solidity > 0.3 and 0.2 < height_ratio < 1.0:
                # print(f"Tọa độ của contour: ({x}, {y}, {w}, {h}), diện tích của vật thể: {cv2.contourArea(contour)}, diện tích của hộp giới hạn {w * h}")
                bounding_rects.append((x, y, w, h))
                character = np.array(mask[y-2: y + h+4, x-2: x + w+4])

                # character = np.array(mask[y - 5: y + h + 5, x - 5:x + w + 5])
                if character.size != 0:
                    # Đảm bảo kích thước ảnh phù hợp với mô hình
                    character_resized = cv2.resize(character, (30, 40))
                    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
                    character_normalized = character_resized / 255.0
                    # Mở rộng chiều dữ liệu để phù hợp với input_shape của mô hình (32, 32, 1)
                    character_input = np.expand_dims(character_normalized, axis=-1)
                    # stat.append(stats[label])
                    candidates.append((character_normalized, (x, y)))

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

    cv2.rectangle(img, (int(X), int(Y)), (int(X + W), int(Y + H)), (0, 0, 255, 2))
    cv2.putText(img, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    path_save = r"D:\HocKy2-23_24\PBL5\Result Test Model V6"
    filename = os.path.basename(img_path)
    save_path = os.path.join(path_save, filename)

    # Lưu ảnh đã xử lý
    cv2.imwrite(save_path, img)
    print(f"Image saved to {save_path}")

for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(file_path)
            img = cv2.imread(file_path)
            if img is not None:
                recognition_license_plate(img, file_path, my_model, model_yolo)
            else:
                print(f"Failed to load image {file_path}")







