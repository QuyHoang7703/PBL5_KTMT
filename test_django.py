import cv2
from django.http import StreamingHttpResponse, HttpResponse
from django.views.decorators import gzip
import torch
from ultralytics import YOLO
from django.urls import path
# Load YOLO model
model = YOLO(r"D:\train_new_2\train_new_2\weights\best.pt")

# Function to stream video
def stream_video(camera, model):
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Process the frame here (e.g., detect objects)
        results = model.predict(frame, augment=True)
        print("alo")
        for detection in results.pred:
            label = detection.names[0]  # Tên của đối tượng được phát hiện
            confidence = detection.conf  # Độ tin cậy của việc phát hiện
            bbox = detection.xyxy[0]  # Tọa độ của bounding box (x_min, y_min, x_max, y_max)

            # Vẽ bounding box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

            # Hiển thị nhãn và độ tin cậy
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lưu ảnh xuống đĩa
        cv2.imwrite("detected_frame.jpg", frame)

        # Yield the frame in the HTTP response
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                                                             b'alo\r\n')


# Decorator to gzip compress the response
@gzip.gzip_page
def video_feed(request):
    try:
        camera = cv2.VideoCapture(0)  # You can change this to read from a video file
        return StreamingHttpResponse(stream_video(camera), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(f"An error occurred: {e}")
        return HttpResponse("An error occurred.", status=500)

# URL pattern for streaming video
urlpatterns = [
    path('video_feed/', video_feed, name='video_feed'),
    # Add other URL patterns as needed
]
