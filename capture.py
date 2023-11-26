import cv2

class ScreenCapture:

    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)

    def capture(self):

        #カメラからの画像取得
        ret, frame = self.cap.read()

        return frame

    def close(self):
        self.cap.release()
