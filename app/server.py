"""
FastAPI Server - WebRTC 串流 API 伺服器
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .worker_manager import WorkerManager
from .webrtc_service import WebRTCService
from .config import (
    ServerConfig,
    ModelConfig,
    get_rtsp_url,
    print_config_summary,
)

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# RTSP 設定（使用配置文件）
RTSP_URL = get_rtsp_url()

# 全域變數
worker_manager: WorkerManager = None
webrtc_service: WebRTCService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    global worker_manager, webrtc_service
    
    # 啟動時初始化
    logger.info("初始化 WorkerManager 與 WebRTC Service...")
    
    # 打印配置摘要
    print_config_summary()
    
    worker_manager = WorkerManager(
        rtsp_url=RTSP_URL,
        use_pose=ModelConfig.USE_POSE,
        use_hands=ModelConfig.USE_HANDS,
        use_yolo=ModelConfig.USE_YOLO,
        device=ModelConfig.DEVICE,
        near_threshold=None,  # 使用配置文件默認值
        max_hands=ModelConfig.MAX_HANDS,
    )
    webrtc_service = WebRTCService(worker_manager)
    logger.info("初始化完成，API server 已就緒")
    
    yield
    
    # 關閉時清理
    logger.info("關閉 API server...")
    if webrtc_service:
        await webrtc_service.close_all_connections()
    if worker_manager:
        worker_manager.stop_worker()
    logger.info("清理完成")


# 建立 FastAPI 應用程式
app = FastAPI(
    title="TH IPCAM Demo API",
    description="產線工作站影像識別 WebRTC 串流服務",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 設定（允許前端跨域存取）
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.ALLOW_ORIGINS,
    allow_credentials=ServerConfig.ALLOW_CREDENTIALS,
    allow_methods=ServerConfig.ALLOW_METHODS,
    allow_headers=ServerConfig.ALLOW_HEADERS,
)


# ===== Pydantic 模型 =====

class SessionStartRequest(BaseModel):
    """啟動 session 請求"""
    use_pose: bool = False
    use_hands: bool = True
    use_yolo: bool = True


class SessionStartResponse(BaseModel):
    """啟動 session 回應"""
    success: bool
    message: str


class SessionStatusResponse(BaseModel):
    """Session 狀態回應"""
    worker_running: bool # VideoWorker 是否在執行
    viewer_count: int    # 目前 viewer 數量
    idle_seconds: int    # 距離最後一位 viewer 斷線的秒數
    webrtc_connections: int # 目前 WebRTC 連線數量
    error: str = None


class SessionStopResponse(BaseModel):
    """停止 session 回應"""
    success: bool
    message: str


class WebRTCOfferRequest(BaseModel):
    """WebRTC offer 請求"""
    sdp: str
    type: str


class WebRTCOfferResponse(BaseModel):
    """WebRTC offer 回應"""
    sdp: str
    type: str


# ===== API Endpoints =====

@app.get("/")
async def root():
    """根路由"""
    return {
        "service": "TH IPCAM Demo API",
        "version": "1.0.0",
        "status": "running",
    }


@app.post("/api/v1/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """
    啟動影像處理 session
    
    此 endpoint 會啟動 VideoWorker，開始處理影像。
    """
    try:
        logger.info(f"收到 start session 請求: {request}")
        
        if worker_manager.is_worker_running():
            return SessionStartResponse(
                success=True,
                message="Session 已在執行中",
            )
        
        # 啟動 worker
        worker_manager.start_worker()
        
        return SessionStartResponse(
            success=True,
            message="Session 已啟動",
        )
        
    except Exception as e:
        logger.exception(f"啟動 session 失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/session/status", response_model=SessionStatusResponse)
async def get_session_status():
    """
    取得目前 session 狀態
    
    回傳 worker 執行狀態、viewer 數量、idle 時間等資訊。
    """
    try:
        status = worker_manager.get_status()
        status["webrtc_connections"] = webrtc_service.get_connection_count()
        
        return SessionStatusResponse(**status)
        
    except Exception as e:
        logger.exception(f"取得 session 狀態失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/session/stop", response_model=SessionStopResponse)
async def stop_session():
    """
    停止影像處理 session
    
    此 endpoint 會停止 VideoWorker 並關閉所有 WebRTC 連線。
    """
    try:
        logger.info("收到 stop session 請求")
        
        # 關閉所有 WebRTC 連線
        await webrtc_service.close_all_connections()
        
        # 停止 worker
        worker_manager.stop_worker()
        
        return SessionStopResponse(
            success=True,
            message="Session 已停止",
        )
        
    except Exception as e:
        logger.exception(f"停止 session 失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/webrtc/offer", response_model=WebRTCOfferResponse)
async def handle_webrtc_offer(request: WebRTCOfferRequest):
    """
    處理 WebRTC offer 並回傳 answer
    
    前端發送 WebRTC offer，後端回傳 answer 以建立連線。
    此 endpoint 會自動啟動 worker（如果尚未啟動）。
    """
    try:
        logger.info("收到 WebRTC offer 請求")
        
        # 如果 worker 未啟動，先啟動
        if not worker_manager.is_worker_running():
            logger.info("Worker 未執行，先啟動...")
            worker_manager.start_worker()
        
        # 處理 offer 並產生 answer
        answer = await webrtc_service.create_offer_response(
            offer_sdp=request.sdp,
            offer_type=request.type,
        )
        
        logger.info("WebRTC answer 已產生")
        
        return WebRTCOfferResponse(**answer)
        
    except Exception as e:
        logger.exception(f"處理 WebRTC offer 失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 啟動伺服器 =====

if __name__ == "__main__":
    import uvicorn
    
    # 啟動 FastAPI server
    uvicorn.run(
        "app.server:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=False,
        log_level="info",
    )
