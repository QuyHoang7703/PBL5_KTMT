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
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")


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

# model = YOLO(r"D:\PyCharm-Project\pythonProject12\runs\detect\train_3\weights\best.pt")
model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
img = cv2.imread(r"D:\HocKy2-23_24\PBL5\Test Model\Screenshot 2024-05-02 145936.png")
results = model.predict(img)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]

# Đọc ảnh biển số đã cắt
X, Y, W, H, X2, Y2 = None, None, None, None, None, None
for result in results:
    boxes = result.boxes.cpu().numpy()
    # print(boxes)
    for box in boxes:
        X = box.xyxy[0][0]
        Y = box.xyxy[0][1]
        # X2 = box.xyxy[0][2]
        # Y2 = box.xyxy[0][3]
        W = box.xywh[0][2]
        H = box.xywh[0][3]

# img_crop = img[int(Y): int(Y)+int(H), int(X): int(X)+int(W)]
# img_crop = img[int(Y)-15: int(Y)+int(H)+15, int(X)-15: int(X)+int(W)+15]
X = int(X)
Y = int(Y)
W = int(W)
H = int(H)

# img_crop = img[Y: Y + H , X : X + W ]
img_crop = img[Y - 2: Y + H + 4, X - 2: X + W + 4]
img_crop = imutils.resize(img_crop, width=390)
cv2.imshow("Before", img_crop)
cv2.waitKey(0)
A, B = find_bottom_points(img_crop)
img_crop = rotate_image(img_crop, A, B)
img_crop = imutils.resize(img_crop, width=390)
cv2.imshow("Before", img_crop)
cv2.waitKey(0)


# img_crop = correct_tilt(img_crop)
# cv2.imshow("After 1.2", img_crop)
# cv2.waitKey(0)

V = cv2.split(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 35, offset=13, method="gaussian")
thresh = (V > T).astype("uint8") * 255

# Hiển thị kết quả

cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
thresh = cv2.bitwise_not(thresh)
thresh = imutils.resize(thresh, width=390)
thresh = cv2.medianBlur(thresh, 5)
cv2.imshow("Thresh after", thresh)
cv2.waitKey(0)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Opened Thresh", thresh)
# cv2.waitKey(0)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Closed after Opened", thresh)
# cv2.waitKey(0)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
#
# # Hiển thị kết quả sau khi điều chỉnh
# cv2.imshow("Thresh after dilation", thresh)
# cv2.waitKey(0)
candidates = []
bounding_rects = []
stat = []
# labels = measure.label(thresh, connectivity=2, background=0)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
# areas = stats[:, cv2.CC_STAT_AREA]
print(len(stats))
print("Kích thước của thresh: ", thresh.shape)
print("Kích thước của thresh: ", thresh.shape[0])
# In ra các diện tích của từng thành phần liên thông
# for i, area in enumerate(areas):
#     print(f"Component {i}: Area = {area}")

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
       if 0.2 < aspect_ratio < 1.0 and 1.0 > solidity > 0.31 and 0.2 < height_ratio < 1.0:
           print(f"Tọa độ của contour: ({x}, {y}, {w}, {h}), diện tích của vật thể: {cv2.contourArea(contour)}, diện tích của hộp giới hạn {w*h}")
           bounding_rects.append((x, y, w, h))
           # character = np.array(img_gray[y-1: y + h+1, x-1: x + w+1])
           # character = img_gray[y - 3: y + h + 3, x - int(h * 3 / 10 - w / 2):x + w + int(h * 3 / 10 - w / 2)]

           character = np.array(mask[y-2: y + h+4, x-2: x + w+4])
           # character = data_until.convert2Square(character)

           # Trích xuất ký tự
           if character.size != 0:
               # Đảm bảo kích thước ảnh phù hợp với mô hình
               character_resized = cv2.resize(character, (30, 40))
               # Chuẩn hóa giá trị pixel về khoảng [0, 1]
               character_normalized = character_resized / 255.0
               # Mở rộng chiều dữ liệu để phù hợp với input_shape của mô hình (32, 32, 1)
               character_input = np.expand_dims(character_normalized, axis=-1)
               stat.append(stats[label])
               candidates.append((character_normalized, (x, y)))
# Load mô hình nhận dạng ký tự
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40_v13_ket_hop.keras")
# my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\MODEL\model_gray_thresh_30_40.keras")

# print(candidates[0])
n = len(candidates)
figure = plt.figure(figsize=(n, 1))

# Vòng lặp để vẽ từng ký tự
for i, (character_input, coords) in enumerate(candidates, 1):
    print(character_input.shape)
    ax = figure.add_subplot(1, n, i)
    ax.imshow(character_input, cmap='gray')  # Hiển thị ảnh xám
    ax.axis('off')  # Tắt trục

plt.show()
# Dự đoán các ký tự từ danh sách các ký tự ứng viên
predicted_characters = []
formatted_candidates = []
for character_input, coords in candidates:
    character_input = np.expand_dims(character_input, axis=0)
    prediction = my_model.predict(character_input)
    # prediction = my_model.predict(np.array(character_input))  # Đảm bảo dữ liệu đầu vào đúng cho mô hình
    print(character_input.shape)
    predicted_index = np.argmax(prediction)
    predicted_character = classes[predicted_index]
    predicted_characters.append(predicted_character)
    formatted_candidates.append((predicted_character, coords))  # Lưu ký tự và tọa độ

# Sắp xếp và định dạng biển số xe
bien_so_du_doan = format_license_plate(formatted_candidates)
print(bien_so_du_doan)

cv2.rectangle(img, (int(X), int(Y)), (int(X + W), int(Y + H)), (0, 0, 255, 2))
cv2.putText(img, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Segmentation", img)
cv2.waitKey()
cv2.destroyAllWindows()
print(len(bounding_rects))
for s in stat:
    print(f"Left: {s[cv2.CC_STAT_LEFT]}, Top: {s[cv2.CC_STAT_TOP]}, Width: {s[cv2.CC_STAT_WIDTH]}, Height: {s[cv2.CC_STAT_HEIGHT]}, Area: {s[cv2.CC_STAT_AREA]}")