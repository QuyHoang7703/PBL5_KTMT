import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from ultralytics import YOLO
import imutils
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

def format(predicted_characters, bounding_rects):
    if len(bounding_rects) == 8:  # Nếu chỉ có một dòng
        # Sắp xếp các ký tự theo tọa độ x của bounding box
        sorted_characters = sorted(zip(predicted_characters, bounding_rects), key=lambda x: x[1][0])
        license_plate = "".join([char[0] for char, _ in sorted_characters])
    else:
        # Phân chia các ký tự thành hai dòng
        first_line = []
        second_line = []
        mid_y = bounding_rects[0][1] + bounding_rects[0][3] / 2

        for character, coordinate in zip(predicted_characters, bounding_rects):
            if coordinate[1] < mid_y:
                first_line.append((character, coordinate[0]))
            else:
                second_line.append((character, coordinate[0]))

        # Sắp xếp các ký tự trong mỗi dòng theo tọa độ x của bounding box
        first_line = sorted(first_line, key=lambda ele: ele[1])
        second_line = sorted(second_line, key=lambda ele: ele[1])

        # Tạo biển số xe từ hai dòng ký tự đã phân loại
        license_plate = "".join([char[0] for char, _ in first_line])  + "".join([char[0] for char, _ in second_line])

    return license_plate


def crop_plate(image):
    # Chuyển ảnh sang xám và tìm contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Giả định rằng biển số là contour lớn nhất
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Tính toán bounding box và cắt ảnh
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


model = YOLO(r"D:\PyCharm-Project\pythonProject12\runs\detect\train_3\weights\best.pt" )
img = cv2.imread(r"D:\test\Screenshot 2024-04-13 153417.png")
results = model.predict(img)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# Đọc ảnh biển số đã cắt
X, Y, W, H = None, None, None, None
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        X = box.xyxy[0][0]
        Y = box.xyxy[0][1]
        W = box.xywh[0][2]
        H = box.xywh[0][3]
# img = cv2.imread('c2.png', cv2.IMREAD_GRAYSCALE)
img_crop = img[int(Y)-5: int(Y)+int(H)+5, int(X)-5: int(X)+int(W)+5]
cv2.imshow("Before", img_crop)
cv2.waitKey(0)

# img_crop = correct_tilt(img_crop)
# cv2.imshow("After 1", img_crop)
# cv2.waitKey(0)
#
# img_crop = crop_plate(img_crop)
# cv2.waitKey(0)
# cv2.imshow("After 2", img_crop)
img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
# Áp dụng ngưỡng thích nghi
binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

# Chuyển đổi ảnh cắt sang HSV và áp dụng ngưỡng địa phương
# img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
# cv2.imshow("img_hsv", img_hsv)
# cv2.waitKey(0)
# V = img_hsv[:, :, 2]  # Trích xuất kênh V
# T = threshold_local(V, 15, offset=10, method="gaussian")
# thresh = (V > T).astype("uint8") * 255

# Hiển thị kết quả

# cv2.imshow("Binary", binary)
# cv2.imshow("Thresh", binary)
# cv2.waitKey(0)
#
# thresh = cv2.bitwise_not(binary)
# thresh = imutils.resize(thresh, width=300)
# thresh = cv2.medianBlur(thresh, 7)
# cv2.imshow("Thresh after", thresh)
# # cv2.imshow("Edged", edged)
# cv2.waitKey(0)


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
# _, labels = cv2.connectedComponents(binary)
print("Num_labels:", num_labels)
# Khởi tạo danh sách để lưu các ký tự ứng viên và bounding rectangles
candidates = []
bounding_rects = []

# Lặp qua các nhãn từ 1 đến num_labels - 1 (loại bỏ nhãn của background)
for label in range(1, num_labels):
    # Tạo mask chứa các pixel có nhãn cùng là label
    mask = np.zeros(binary.shape, dtype=np.uint8)
    mask[labels == label] = 255 # Các các pixel cùng nhãn giá trị 255

    # Tìm contours từ mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / float(h)
        solidity = cv2.contourArea(contour) / float(w * h)
        height_ratio = h / float(binary.shape[0])

        # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
        if 0.1 < aspect_ratio < 1.0 and 1.0>solidity > 0.1 and 0.2 < height_ratio < 2.0:
            bounding_rects.append((x, y, w, h))
            character = np.array(mask[y-2: y + h+2, x-2:x + w+2])
            # character = np.array(mask[y: y + h, x :x + w])
            # Trích xuất ký tự
            if character.size != 0:

                # Đảm bảo kích thước ảnh phù hợp với mô hình
                character_resized = cv2.resize(character, (30, 40), interpolation=cv2.INTER_AREA)
                # Chuẩn hóa giá trị pixel về khoảng [0, 1]
                character_normalized = character_resized / 255.0
                # Mở rộng chiều dữ liệu để phù hợp với input_shape của mô hình (32, 32, 1)
                character_input = np.expand_dims(character_normalized, axis=-1)
                # Thêm ký tự đã chuẩn bị vào danh sách các ký tự
                # candidates.append(character_input)
                candidates.append(character_input)


for rect in bounding_rects:
    x, y, w, h = rect
    cv2.rectangle(img_crop, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Load mô hình nhận dạng ký tự
my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT\model_gray_thresh_30_40.h5")

n = len(candidates)
figure = plt.figure(figsize=(n, 1))

# Vòng lặp để vẽ từng ký tự
for i, character in enumerate(candidates, 1):
    ax = figure.add_subplot(1, n, i)
    ax.imshow(character, cmap='gray')  # Hiển thị ảnh xám
    ax.axis('off')  # Tắt trục

plt.show()
# Dự đoán các ký tự từ danh sách các ký tự ứng viên
predicted_characters = []
for character_input in candidates:
    prediction = my_model.predict(np.array([character_input]))
    # Lấy chỉ số của lớp có xác suất cao nhất
    predicted_index = np.argmax(prediction)
    # Chuyển chỉ số thành ký tự dự đoán
    predicted_character = classes[predicted_index]  # Sử dụng danh sách classes để ánh xạ chỉ số sang ký tự
    # Thêm ký tự dự đoán vào danh sách
    predicted_characters.append(predicted_character)

bien_so_du_doan = format(predicted_characters, bounding_rects)

print("Biển số xe được dự đoán:", bien_so_du_doan)

cv2.rectangle(img, (int(X), int(Y)), (int(X+W), int(Y+H)), (0, 0, 255, 2))
cv2.putText(img, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Segmentation", img)
cv2.waitKey()
cv2.destroyAllWindows()
print(len(bounding_rects))