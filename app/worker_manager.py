"""
WorkerManager - 管理 VideoWorker 生命週期
負責啟停、viewer_count 追蹤、15 分鐘 idle timeout
"""
import asyncio
import logging
import threading
import time
from typing import Optional

from .video_worker import VideoWorker

logger = logging.getLogger(__name__)


class WorkerManager:
    """管理 VideoWorker 啟停與 idle timeout"""
    
    IDLE_TIMEOUT_SECONDS = 1 * 60  # 1 分鐘
    
    def __init__(
        self,
        rtsp_url: str,
        use_pose: bool = False,
        use_hands: bool = True,
        use_yolo: bool = True,
        device: Optional[str] = None,
        near_threshold: int = 50,
        max_hands: int = 4,
    ):
        """
        初始化 WorkerManager
        
        Args:
            rtsp_url: RTSP 網址
            use_pose: 是否啟用 YOLOv8-pose
            use_hands: 是否啟用 MediaPipe Hands
            use_yolo: 是否啟用 YOLO 物體偵測
            device: 推論裝置
            near_threshold: 手部附近物體閾值
            max_hands: 同時偵測的手部數量
        """
        self._worker: Optional[VideoWorker] = None
        self._worker_config = {
            "rtsp_url": rtsp_url,
            "use_pose": use_pose,
            "use_hands": use_hands,
            "use_yolo": use_yolo,
            "device": device,
            "near_threshold": near_threshold,
            "max_hands": max_hands,
        }
        
        # Viewer 計數與 idle timeout
        self._viewer_count = 0
        self._viewer_lock = threading.Lock()
        self._last_viewer_time = 0.0
        
        # Idle timeout 監控執行緒
        self._timeout_thread: Optional[threading.Thread] = None
        self._stop_timeout_event = threading.Event()
    
    def start_worker(self):
        """啟動 VideoWorker"""
        if self._worker and self._worker.is_running:
            logger.info("VideoWorker 已在執行中")
            return
        
        # 如果有舊的 worker，確保完全清理
        if self._worker:
            logger.warning("檢測到未清理的 worker，先進行清理...")
            try:
                self._worker.stop()
            except Exception as e:
                logger.error(f"清理舊 worker 失敗: {e}")
            self._worker = None
            time.sleep(0.5)  # 給一點時間讓資源完全釋放
        
        logger.info("啟動 VideoWorker...")
        try:
            self._worker = VideoWorker(**self._worker_config)
            self._worker.start()
            
            # 啟動 idle timeout 監控
            self._start_timeout_monitor()
            
            logger.info("VideoWorker 啟動成功")
        except Exception as e:
            logger.exception(f"啟動 VideoWorker 失敗: {e}")
            self._worker = None
            raise
    
    def stop_worker(self):
        """停止 VideoWorker（外部調用）"""
        if not self._worker:
            return
        
        logger.info("停止 VideoWorker（外部請求）...")
        
        # 停止 timeout 監控
        self._stop_timeout_monitor()
        
        # 停止 worker
        self._worker.stop()
        self._worker = None
        
        # 重置 viewer 計數
        with self._viewer_lock:
            self._viewer_count = 0
            self._last_viewer_time = 0.0
    
    def _stop_worker_internal(self):
        """停止 VideoWorker（內部調用，避免死鎖）"""
        if not self._worker:
            return
        
        logger.info("停止 VideoWorker（timeout 觸發）...")
        
        # 直接停止 worker，不處理 timeout monitor（因為就在 monitor 中）
        self._worker.stop()
        self._worker = None
        
        # 重置 viewer 計數
        with self._viewer_lock:
            self._viewer_count = 0
            self._last_viewer_time = 0.0
    
    def add_viewer(self):
        """增加 viewer 計數"""
        should_start_worker = False
        
        with self._viewer_lock:
            self._viewer_count += 1
            self._last_viewer_time = time.time()
            logger.info(f"Viewer 加入，目前數量: {self._viewer_count}")
            
            # 如果是第一個 viewer，標記需要啟動 worker
            if self._viewer_count == 1:
                if not self._worker or not self._worker.is_running:
                    should_start_worker = True
        
        # 在鎖外部啟動 worker（避免死鎖）
        if should_start_worker:
            logger.info("檢測到第一個 viewer，啟動 worker...")
            try:
                self.start_worker()
            except Exception as e:
                logger.exception(f"啟動 worker 失敗: {e}")
                # 如果啟動失敗，減少 viewer 計數
                with self._viewer_lock:
                    self._viewer_count -= 1
    
    def remove_viewer(self):
        """減少 viewer 計數"""
        with self._viewer_lock:
            if self._viewer_count > 0:
                self._viewer_count -= 1
                logger.info(f"Viewer 離開，目前數量: {self._viewer_count}")
                
                # 當最後一個 viewer 離開時，記錄時間開始計算 idle
                if self._viewer_count == 0:
                    self._last_viewer_time = time.time()
                    logger.info("所有 viewer 已離開，開始計算 idle 時間")
    
    def get_viewer_count(self) -> int:
        """取得目前 viewer 數量"""
        with self._viewer_lock:
            return self._viewer_count
    
    def get_latest_frame(self):
        """取得最新影格"""
        if not self._worker or not self._worker.is_running:
            return None
        return self._worker.get_latest_frame()
    
    def is_worker_running(self) -> bool:
        """檢查 worker 是否正在執行"""
        return self._worker is not None and self._worker.is_running
    
    def get_worker_error(self) -> Optional[str]:
        """取得 worker 錯誤訊息"""
        if self._worker:
            return self._worker.error
        return None
    
    def _start_timeout_monitor(self):
        """啟動 idle timeout 監控執行緒"""
        if self._timeout_thread and self._timeout_thread.is_alive():
            return
        
        self._stop_timeout_event.clear()
        self._timeout_thread = threading.Thread(target=self._timeout_monitor_loop, daemon=True)
        self._timeout_thread.start()
        logger.info("Idle timeout 監控已啟動")
    
    def _stop_timeout_monitor(self):
        """停止 idle timeout 監控執行緒"""
        if not self._timeout_thread:
            return
        
        logger.info("停止 idle timeout 監控...")
        self._stop_timeout_event.set()
        
        if self._timeout_thread.is_alive():
            self._timeout_thread.join(timeout=2.0)
        
        self._timeout_thread = None
    
    def _timeout_monitor_loop(self):
        """Idle timeout 監控迴圈"""
        logger.info("Idle timeout 監控迴圈開始")
        
        while not self._stop_timeout_event.is_set():
            try:
                # 檢查 viewer 數量與最後活動時間
                with self._viewer_lock:
                    viewer_count = self._viewer_count
                    last_time = self._last_viewer_time
                
                # 如果沒有 viewer 且超過 timeout 時間，停止 worker
                if viewer_count == 0 and last_time > 0:
                    idle_duration = time.time() - last_time
                    if idle_duration >= self.IDLE_TIMEOUT_SECONDS:
                        logger.info(f"無 viewer 已持續 {idle_duration:.1f} 秒，停止 worker...")
                        # 避免在自己的執行緒中 join 自己，設定標記後由外部處理
                        self._stop_worker_internal()
                        break
                
                # 每 10 秒檢查一次
                self._stop_timeout_event.wait(timeout=10.0)
                
            except Exception as e:
                logger.exception(f"Timeout 監控錯誤: {e}")
                time.sleep(1.0)
        
        logger.info("Idle timeout 監控迴圈結束")
    
    def get_status(self) -> dict:
        """取得目前狀態"""
        with self._viewer_lock:
            viewer_count = self._viewer_count
            last_time = self._last_viewer_time
        
        status = {
            "worker_running": self.is_worker_running(),
            "viewer_count": viewer_count,
            "idle_seconds": 0,
        }
        
        if viewer_count == 0 and last_time > 0:
            status["idle_seconds"] = int(time.time() - last_time)
        
        error = self.get_worker_error()
        if error:
            status["error"] = error
        
        return status
