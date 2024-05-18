import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
if torch.cuda.is_available():
    print("Using CUDA")
else:
    print("Using CPU")

# Load dữ liệu từ file numpy đã lưu trước đó
images = np.load(r"DATA/images_gray_thresh_30_40_ket_hop.npy")
labels = np.load(r"DATA/labels_gray_thresh_30_40_ket_hop.npy")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_images, valid_images, train_labels, valid_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu ảnh về khoảng [0, 1]
train_images = train_images.astype('float32') / 255.0
valid_images = valid_images.astype('float32') / 255.0

# One-hot encode nhãn
train_labels = tf.keras.utils.to_categorical(train_labels, 30)
valid_labels = tf.keras.utils.to_categorical(valid_labels, 30)
import tensorflow as tf
from tensorflow.keras import layers, models


data_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    # shear_range=0.2,
    zoom_range=0.15,
    # blur_range=[3, 7],
    # brightness_range=[0.6, 1.2],
    fill_mode='nearest'
)
train_generator = data_augmentation.flow(train_images, train_labels, batch_size=64)
valid_generator = data_augmentation.flow(valid_images, valid_labels, batch_size = 64)
def alexnet_custom_small(input_shape, number_classes):
    model = Sequential()

    # Adjusted 1st Convolutional Layer with increased filters
    model.add(Conv2D(filters=32, input_shape=input_shape, kernel_size=(3,3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Dropout(0.25))

    # 2nd Convolutional Layer with increased filters
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Dropout(0.25))

    # 3rd Convolutional Layer with increased filters
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 4th Convolutional Layer with increased filters
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 5th Convolutional Layer with increased filters
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Final Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Dropout(0.25))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(128))  # Increased dense layer size
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(128))  # Increased dense layer size
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(number_classes))
    model.add(Activation('softmax'))

    return model
# Tạo mô hình
model = alexnet_custom_small((40, 30, 1), 30)
model.summary()
checkpoint_callback = ModelCheckpoint(
    'MODEL/model_gray_thresh_30_40_v14_ket_hop.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
)
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=5,
    verbose=1
)
# model = modified_alexnet_multiclass((32 ,32, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    epochs=60,
    # batch_size=32,
    validation_data= valid_generator,
    callbacks=[checkpoint_callback, reduce_lr_callback],
    verbose=1
)
plt.figure(figsize=(12, 5))

# Vẽ đồ thị độ chính xác
plt.subplot(1, 2, 1)  # subplot với 1 hàng, 2 cột, vị trí thứ 1
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Vẽ đồ thị mất mát
plt.subplot(1, 2, 2)  # subplot với 1 hàng, 2 cột, vị trí thứ 2
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Hiển thị hình vẽ
plt.tight_layout()
plt.show()