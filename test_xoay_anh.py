import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from ultralytics import YOLO
import imutils
from imutils import perspective
import data_until
from imutils.perspective import four_point_transform
from skimage import measure
import math
import torch
import os


def maximizeContrast(imgGrayscale):
    # Làm cho độ tương phản lớn nhất
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # tạo bộ lọc kernel

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement,
                                 iterations=10)  # nổi bật chi tiết sáng trong nền tối
    # cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement,
                                   iterations=10)  # Nổi bật chi tiết tối trong nền sáng
    # cv2.imwrite("blackhat.jpg",imgBlackHat)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    # Kết quả cuối là ảnh đã tăng độ tương phản
    return imgGrayscalePlusTopHatMinusBlackHat

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
img = cv2.imread(r"D:\HocKy2-23_24\PBL5\Test Model\Screenshot 2024-04-30 150300.png")
results = model.predict(img)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
path = r"D:\HocKy2-23_24\PBL5\Test Model"
model_yolo = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
path_save = r"D:\HocKy2-23_24\PBL5\Result Rotate Image"


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
def crop_and_rotate(img, img_path, model_yolo):
    X, Y, W, H, X2, Y2 = None, None, None, None, None, None
    results = model_yolo.predict(img)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            X = box.xyxy[0][0]
            Y = box.xyxy[0][1]
            W = box.xywh[0][2]
            H = box.xywh[0][3]
    X = int(X)
    Y = int(Y)
    W = int(W)
    H = int(H)
    # img_crop = img[Y - 5: Y + H + 10, X - 5: X + W + 10]
    img_crop = img[int(Y) - 2: int(Y) + int(H) + 4, int(X) - 2: int(X) + int(W) + 4]
    A, B = find_bottom_points(img_crop)
    img_rotate = rotate_image(img_crop, A, B)

    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(path_save, filename)
    save_path_rotate = os.path.join(path_save, f"{name}_rotate{ext}")

    # Lưu ảnh đã cắt
    cv2.imwrite(save_path, img_crop)
    print(f"Image saved to {save_path}")

    # Lưu ảnh đã xoay
    cv2.imwrite(save_path_rotate, img_rotate)
    print(f"Rotated image saved to {save_path_rotate}")


for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(file_path)
            img = cv2.imread(file_path)
            if img is not None:
                crop_and_rotate(img, file_path, model_yolo)
            else:
                print(f"Failed to load image {file_path}")