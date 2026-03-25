"""
Webcam 影像擷取模組
- 以 OpenCV 開啟本機攝影機（Webcam）
- 介面與 RTSPCapture 一致：read() 回傳 (ret, frame)、release() 釋放資源
"""
import threading
import cv2


class WebcamCapture:
    def __init__(self, camera_id=0):
        """
        Args:
            camera_id (int): 攝影機編號，0 為預設、1 為第二台…依此類推。
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = None
        self.stopped = False
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.q = frame

    def read(self):
        """回傳 (ret, frame)，與 cv2.VideoCapture.read() 用法一致。"""
        return self.q is not None, self.q

    def release(self):
        """停止背景執行緒並釋放攝影機。"""
        self.stopped = True
        self.cap.release()
