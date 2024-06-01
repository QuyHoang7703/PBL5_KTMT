import cv2

import urllib.request
import numpy as np

url = 'http://10.10.58.64'

while True:
    # Lấy hình ảnh từ URL
    img_req = urllib.request.urlopen(url)
    # Chuyển đổi hình ảnh thành mảng numpy
    img_np = np.array(bytearray(img_req.read()), dtype=np.uint8)
    # Giải mã mảng numpy thành hình ảnh
    img_frame = cv2.imdecode(img_np, -1)
    
    if img_frame is not None:
        # Hiển thị hình ảnh
        cv2.imshow("Nhan dien bien so xe", img_frame)
        
        # Thoát khỏi vòng lặp khi nhấn phím 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        print("Không tải được hình ảnh")

# Đóng tất cả các cửa sổ

cv2.destroyAllWindows()
