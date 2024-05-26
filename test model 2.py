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
def process_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(img_gray, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    # cv2.imshow("avb", dilated_image)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            [x, y, w, h] = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 2 <= ratio <= 5:  # Consider typical license plate aspect ratio
                screenCnt = approx
                break

    if screenCnt is None:
        print("No plate detected")
        return img

    # Sort the points in the contour based on their y-coordinates (top to bottom)
    sorted_points = sorted(screenCnt, key=lambda x: x[0][1])

    # Extract the bottom two points
    bottom_points = sorted_points[-2:]
    bottom_left = min(bottom_points, key=lambda x: x[0][0])
    bottom_right = max(bottom_points, key=lambda x: x[0][0])

    # Calculate the angle to rotate
    delta_y = bottom_right[0][1] - bottom_left[0][1]
    delta_x = bottom_right[0][0] - bottom_left[0][0]
    angle = math.atan2(delta_y, delta_x) * (180.0 / math.pi)
    center = (int((bottom_left[0][0] + bottom_right[0][0]) / 2), int((bottom_left[0][1] + bottom_right[0][1]) / 2))

    # Rotate the image
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return rotated
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
        img_crop = img[int(Y) - 2: int(Y) + int(H) + 4, int(X) - 2: int(X) + int(W) + 4]
    else:
        print("No license plate detected.")
        return

    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
    #                                int(W / 20) if int(W / 20) % 2 != 0 else int(W / 20) + 1, 15)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                   15, 5)
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
                character = np.array(mask[y-3: y + h+6, x-3: x + w+6])

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







