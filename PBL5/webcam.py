import cv2

class Webcam():
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
        
    def get_frame(self):
        if not self.vid.isOpened(): 
            return
        while True:
            _,img = self.vid.read()
            yield cv2.imencode('.jpg',img)[1].tobytes()