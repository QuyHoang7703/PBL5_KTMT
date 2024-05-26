import cv2
from ultralytics import YOLO
model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")
vid = cv2.VideoCapture("http://192.168.1.16:81/stream")
import pandas
while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        print("Không thể nhận frame")
        # break
    else:
        # frame = cv2.resize(frame, (800, 600))
        # results = model(frame)
        # list_plates = results.pandas().xyxy[0].values.tolist()
        # for plate in list_plates:
        #     flag = 0
        #     x = int(plate[0])  # xmin
        #     y = int(plate[1])  # ymin
        #     w = int(plate[2] - plate[0])  # xmax - xmin
        #     h = int(plate[3] - plate[1])  # ymax - ymin
        #     cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), color=(0, 0, 225), thickness=2)
        cv2.imshow("Segmentation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

vid.release()
cv2.destroyAllWindows()