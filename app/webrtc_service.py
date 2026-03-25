"""
WebRTC Service - 處理 WebRTC 連線與影像串流
"""
import asyncio
import logging
from typing import Optional

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame

logger = logging.getLogger(__name__)


class ProcessedVideoTrack(VideoStreamTrack):
    """從 WorkerManager 取得處理後的影格並透過 WebRTC 串流"""
    
    kind = "video"
    
    def __init__(self, worker_manager):
        """
        初始化 ProcessedVideoTrack
        
        Args:
            worker_manager: WorkerManager 實例
        """
        super().__init__()
        self.worker_manager = worker_manager
        self._counter = 0
        self._last_log_time = 0
        logger.info(f"ProcessedVideoTrack 已建立")
    
    async def recv(self):
        """接收並回傳下一個影格"""
        pts, time_base = await self.next_timestamp()
        
        # 從 WorkerManager 取得最新影格
        frame = self.worker_manager.get_latest_frame()
        
        if frame is None:
            # 如果沒有影格，回傳黑色畫面
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Waiting for video...",
                (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            # 每 30 幀記錄一次（約每秒一次）
            if self._counter % 30 == 0:
                logger.warning(f"ProcessedVideoTrack: 等待影格... (counter={self._counter})")
        
        # 轉換 BGR 到 RGB（OpenCV 使用 BGR，WebRTC 需要 RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 建立 VideoFrame
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        self._counter += 1
        
        # 每 60 幀記錄一次（約每 2 秒）
        import time
        current_time = time.time()
        if current_time - self._last_log_time >= 2.0:
            logger.info(f"ProcessedVideoTrack: 已傳送 {self._counter} 幀，解析度: {frame.shape[1]}x{frame.shape[0]}")
            self._last_log_time = current_time
        
        return video_frame


class WebRTCService:
    """管理 WebRTC 連線"""
    
    def __init__(self, worker_manager):
        """
        初始化 WebRTC Service
        
        Args:
            worker_manager: WorkerManager 實例
        """
        self.worker_manager = worker_manager
        self.pcs = set()  # 儲存所有 RTCPeerConnection
    
    async def create_offer_response(self, offer_sdp: str, offer_type: str) -> dict:
        """
        處理 WebRTC offer 並產生 answer
        
        Args:
            offer_sdp: Offer SDP 內容
            offer_type: Offer 類型（通常是 "offer"）
        
        Returns:
            包含 answer SDP 的字典
        """
        # 建立 RTCPeerConnection（配置 ICE 使用本地 IP）
        from aiortc.contrib.media import MediaRecorder
        import os
        
        # 嘗試取得主要的本地 IP（優先使用 192.168.0.156）
        ice_servers = []
        
        # 建立 configuration
        configuration = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(configuration=configuration)
        
        self.pcs.add(pc)
        
        pc_id = id(pc)
        logger.info(f"[{pc_id}] 建立 WebRTC 連線（內網模式），目前連線數: {len(self.pcs)}")
        
        # 增加 viewer 計數
        self.worker_manager.add_viewer()
        
        # 當連線狀態改變時的處理
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"[{pc_id}] WebRTC 連線狀態: {pc.connectionState}")
            if pc.connectionState == "connected":
                logger.info(f"[{pc_id}] ✅ WebRTC 連線已成功建立")
            elif pc.connectionState in ["failed", "closed"]:
                logger.warning(f"[{pc_id}] WebRTC 連線已關閉或失敗")
                await self.close_peer_connection(pc)
        
        # ICE 連線狀態監控
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"[{pc_id}] ICE 連線狀態: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                logger.error(f"[{pc_id}] ❌ ICE 連線失敗（網路協商失敗）")
            elif pc.iceConnectionState == "disconnected":
                logger.warning(f"[{pc_id}] ICE 連線中斷")
        
        # ICE gathering 狀態監控
        @pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            logger.info(f"[{pc_id}] ICE gathering 狀態: {pc.iceGatheringState}")
        
        # 當收到 track 時的處理（通常前端不會傳送 track）
        @pc.on("track")
        def on_track(track):
            logger.info(f"[{pc_id}] 收到 track: {track.kind}")
        
        # 建立影像 track
        logger.info(f"[{pc_id}] 建立 ProcessedVideoTrack...")
        video_track = ProcessedVideoTrack(self.worker_manager)
        pc.addTrack(video_track)
        logger.info(f"[{pc_id}] 已添加視頻軌道")
        
        # 設定 remote description（offer）
        logger.info(f"[{pc_id}] 設定 remote description (offer)...")
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await pc.setRemoteDescription(offer)
        logger.info(f"[{pc_id}] Remote description 已設定")
        
        # 建立 answer
        logger.info(f"[{pc_id}] 建立 answer...")
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        logger.info(f"[{pc_id}] WebRTC answer 已建立並設定為 local description")
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    
    async def close_peer_connection(self, pc: RTCPeerConnection):
        """關閉單一 peer connection"""
        try:
            if pc in self.pcs:
                self.pcs.remove(pc)
                logger.info(f"移除 WebRTC 連線，剩餘連線數: {len(self.pcs)}")
                
                # 減少 viewer 計數
                self.worker_manager.remove_viewer()
            
            await pc.close()
        except Exception as e:
            logger.error(f"關閉 peer connection 錯誤: {e}")
    
    async def close_all_connections(self):
        """關閉所有 WebRTC 連線"""
        logger.info("關閉所有 WebRTC 連線...")
        
        # 複製連線列表以避免在迭代時修改
        pcs_copy = list(self.pcs)
        
        for pc in pcs_copy:
            await self.close_peer_connection(pc)
        
        logger.info("所有 WebRTC 連線已關閉")
    
    def get_connection_count(self) -> int:
        """取得目前連線數量"""
        return len(self.pcs)
