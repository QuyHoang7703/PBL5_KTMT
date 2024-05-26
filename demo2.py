import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from ultralytics import YOLO
import imutils
from skimage.filters import threshold_local
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
        license_plate = "".join([char[0] for char, _ in first_line]) + "-" + "".join([char[0] for char, _ in second_line])

    return license_plate

model = YOLO(r"D:\PyCharm-Project\pythonProject12\runs\detect\train_3\weights\best.pt" )
img = cv2.imread(r"D:\PyCharm-Project\pythonProject12\test\a7.png")
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
img_crop = img[int(Y)-3: int(Y)+int(H)+3, int(X)-3: int(X)+int(W)+3]
# img_crop=correct_tilt(img_crop)
# img_crop_resize = cv2.resize(img_crop,None,  fx=2, fy=2)
cv2.imshow("Crop_img", img_crop)
cv2.waitKey()
# img = cv2.imread('c2.png')
img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
# Chuyển đổi sang ảnh nhị phân
# _, binary = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# binary = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#
# cv2.imshow("binary", binary)
# cv2.waitKey(0)
image = img_crop.copy()
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

# Áp dụng ngưỡng thích nghi
T = threshold_local(V, 35, offset=5, method="gaussian")
thresh = (V > T).astype("uint8") * 255
thresh = cv2.bitwise_not(thresh)
# thresh = imutils.resize(thresh, width=400)

# Sử dụng connected components để tìm và lọc ký tự
num_labels, labels = cv2.connectedComponents(thresh)
mask = np.zeros(thresh.shape, dtype="uint8")
total_pixels = thresh.shape[0] * thresh.shape[1]
lower = total_pixels // 90
upper = total_pixels // 20

for label in range(1, num_labels):  # Bỏ qua nhãn 0 vì đó là nền
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    if lower < numPixels < upper:
        mask = cv2.add(mask, labelMask)

# Tìm contours từ mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in contours]
boundingBoxes = np.array(boundingBoxes)
mean_w = np.mean(boundingBoxes[:, 2])
mean_h = np.mean(boundingBoxes[:, 3])
mean_y = np.mean(boundingBoxes[:, 1])
threshold_w = mean_w * 1.5
threshold_h = mean_h * 1.5

# Lọc bounding boxes dựa trên kích thước
new_boundingBoxes = boundingBoxes[(boundingBoxes[:, 2] < threshold_w) & (boundingBoxes[:, 3] < threshold_h)]

# Phân loại bounding boxes vào hai dòng
line1, line2 = [], []
for box in new_boundingBoxes:
    x, y, w, h = box
    if y > mean_y * 1.2:
        line2.append(box)
    else:
        line1.append(box)

# Sắp xếp các boxes theo thứ tự từ trái sang phải
line1 = sorted(line1, key=lambda box: box[0])
line2 = sorted(line2, key=lambda box: box[0])
boundingBoxes = line1 + line2

# Vẽ bounding boxes lên ảnh
img_with_boxes = imutils.resize(image.copy(), width=400)
# for bbox in boundingBoxes:
#     x, y, w, h = bbox
#     cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Final Result", img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

model_detect_text = load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\model_gray_thresh.h5")

chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'
]

vehicle_plate = ""
characters = []

# Đọc ảnh, giả sử img đã được tải và img_crop đã được xử lý từ trước
image = img_crop.copy()

# Tiếp tục từ các bước trước, giả sử mask và boundingBoxes đã được tính toán
for rect in boundingBoxes:
    x, y, w, h = rect
    character = mask[y:y+h, x:x+w]
    character = cv2.bitwise_not(character)
    rows, columns = character.shape[:2]
    paddingY = (32 - rows) // 2 if rows < 32 else int(0.17 * rows)
    paddingX = (32 - columns) // 2 if columns < 32 else int(0.45 * columns)
    character = cv2.copyMakeBorder(character, paddingY, paddingY,
                                   paddingX, paddingX, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # character = cv2.cvtColor(character, cv2.COLOR_GRAY2RGB)
    character = cv2.resize(character, (32, 32))
    character = character.astype("float") / 255.0
    character = np.expand_dims(character, axis=-1)
    characters.append(character)

characters = np.array(characters)
probs = model_detect_text.predict(characters)

# for prob in probs:
#     idx = np.argmax(prob)
#     vehicle_plate += chars[idx]
for prob in probs:
    idx = np.argmax(prob)
    print(f"Predicted index: {idx}, Probability: {prob[idx]}")
    if idx < len(chars):
        vehicle_plate += chars[idx]
    else:
        print(f"Index out of range: {idx}")

print("Detected License Plate:", vehicle_plate)



















# Áp dụng Median Blur để làm mịn ảnh
# binary_median = cv2.medianBlur(binary, 5)
# binary_median = imutils.resize(binary_median, width=400)
# # Hiển thị các kết quả
# cv2.imshow("Binary After Median Blur", binary_median)
#
# cv2.waitKey(0)
#
# # Tìm connected components và gán nhãn cho từng pixel
# # num_labels: số lượng các thành phần kết nối có trong ảnh
# # labels: tên của các pixel thuộc cùng 1 thành phần liên kết => Các pixel thuộc một vùng kết nối sẽ có cùng label
# # stat: chứa thông tin của thành phần kết nối bao gồm rectangle bao quanh nó (tâm, width, hight), diện tích của thành phần kết nối
# # centroid: chứa tọa độ tâm của rectangle chứa thành phần kết nối
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_median, connectivity=8)
# # _, labels = cv2.connectedComponents(binary)
# print("Num_labels:", num_labels)
# # Khởi tạo danh sách để lưu các ký tự ứng viên và bounding rectangles
# candidates = []
# bounding_rects = []
#
# # Lặp qua các nhãn từ 1 đến num_labels - 1 (loại bỏ nhãn của background)
# for label in range(1, num_labels):
#     # Tạo mask chứa các pixel có nhãn cùng là label
#     mask = np.zeros(binary_median.shape, dtype=np.uint8)
#     mask[labels == label] = 255 # Các các pixel cùng nhãn giá trị 255
#
#     # Tìm contours từ mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#
#         aspect_ratio = w / float(h)
#         solidity = cv2.contourArea(contour) / float(w * h)
#         height_ratio = h / float(binary.shape[0])
#
#         # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
#         if 0.1 < aspect_ratio < 1.0 and 1.0>solidity > 0.1 and 0.2 < height_ratio < 1.0:
#             bounding_rects.append((x, y, w, h))
#             # character = binary[y-3: y + h+3, x-3:x + w+3]
#             character = binary_median[y-3: y + h+3, x-3:x + w+3]
#             # Trích xuất ký tự
#             if character.size != 0:
#
#                 # Đảm bảo kích thước ảnh phù hợp với mô hình
#                 character_resized = cv2.resize(character, (30, 40), interpolation=cv2.INTER_AREA)
#                 # Chuẩn hóa giá trị pixel về khoảng [0, 1]
#                 character_normalized = character_resized / 255.0
#                 # Mở rộng chiều dữ liệu để phù hợp với input_shape của mô hình (32, 32, 1)
#                 character_input = np.expand_dims(character_normalized, axis=-1)
#                 # Thêm ký tự đã chuẩn bị vào danh sách các ký tự
#                 # candidates.append(character_input)
#                 candidates.append(character_input)
#
# for rect in bounding_rects:
#     x, y, w, h = rect
#     cv2.rectangle(img_crop, (x, y), (x+w, y+h), (0, 0, 255), 2)
#
# # Load mô hình nhận dạng ký tự
# my_model = load_model(r"D:\PyCharm-Project\PBL5-KTMT\model_gray_thresh_30_40.h5")
#
# n = len(candidates)
# figure = plt.figure(figsize=(n, 1))
#
# # Vòng lặp để vẽ từng ký tự
# for i, character in enumerate(candidates, 1):
#     ax = figure.add_subplot(1, n, i)
#     ax.imshow(character, cmap='gray')  # Hiển thị ảnh xám
#     ax.axis('off')  # Tắt trục
#
# plt.show()
# # Dự đoán các ký tự từ danh sách các ký tự ứng viên
# predicted_characters = []
# for character_input in candidates:
#     prediction = my_model.predict(np.array([character_input]))
#     # Lấy chỉ số của lớp có xác suất cao nhất
#     predicted_index = np.argmax(prediction)
#     # Chuyển chỉ số thành ký tự dự đoán
#     predicted_character = classes[predicted_index]  # Sử dụng danh sách classes để ánh xạ chỉ số sang ký tự
#     # Thêm ký tự dự đoán vào danh sách
#     predicted_characters.append(predicted_character)
#
# bien_so_du_doan = format(predicted_characters, bounding_rects)
#
# print("Biển số xe được dự đoán:", bien_so_du_doan)
#
# cv2.rectangle(img, (int(X), int(Y)), (int(X+W), int(Y+H)), (0, 0, 255, 2))
# cv2.putText(img, bien_so_du_doan, (int(X), int(Y)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
# cv2.imshow("Segmentation", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
