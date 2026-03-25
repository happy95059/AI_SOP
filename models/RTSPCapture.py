import threading
import cv2
import time

class RTSPCapture:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = None
        self.stopped = False
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                with self.lock:
                    self.q = frame
            except Exception as e:
                if not self.stopped:
                    print(f"RTSPCapture reader error: {e}")
                break

    def read(self):
        with self.lock:
            return self.q is not None, self.q

    def release(self):
        # 先設置停止標記
        self.stopped = True
        
        # 等待讀取執行緒結束（最多等待 2 秒）
        if self.t and self.t.is_alive():
            self.t.join(timeout=2.0)
        
        # 釋放 VideoCapture
        try:
            if self.cap:
                self.cap.release()
        except Exception as e:
            print(f"RTSPCapture release error: {e}")

