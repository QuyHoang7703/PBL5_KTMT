import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Kiểm tra GPU
if tf.config.list_physical_devices('GPU'):
    print("Using CUDA")
else:
    print("Using CPU")
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
# Load dữ liệu từ file numpy đã lưu trước đó
images = np.load(r"images_gray_thresh_32_32_update.npy")
labels = np.load(r"labels_gray_thresh_32_32_update.npy")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=40)

# Chuẩn hóa dữ liệu ảnh về khoảng [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode nhãn
train_labels = tf.keras.utils.to_categorical(train_labels, 30)
test_labels = tf.keras.utils.to_categorical(test_labels, 30)

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model(r"D:\PyCharm-Project\PBL5-KTMT_2\best_model_gray_thresh_32_32_alexnet.keras")

# Dự đoán nhãn cho tập kiểm tra
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Tạo confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Vẽ confusion matrix dưới dạng heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

