#!/usr/bin/env python3
"""
啟動 WebRTC 串流 API Server
"""
import os
import sys
from app import config
# 設定 RTSP URL（可透過環境變數覆蓋）
# 使用 UDP: /udp/av0_0，使用 TCP: /tcp/av0_0
if "RTSP_URL" not in os.environ:
    # 預設使用 TCP（較穩定）
    os.environ["RTSP_URL"] = config.get_rtsp_url()
    
    # 如果需要使用 UDP（較低延遲），取消下面這行的註解
    # os.environ["RTSP_URL"] = "rtsp://admin:50984878@192.168.0.205:10554/udp/av0_0"

# 設定 OpenCV FFMPEG 選項（配合 RTSP 傳輸模式）
if "/tcp/" in os.environ.get("RTSP_URL", ""):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
else:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("TH_IPCAM_DEMO - WebRTC 串流 API Server")
    print("=" * 60)
    print(f"RTSP URL: {os.environ['RTSP_URL']}")
    print(f"API Server: http://0.0.0.0:{config.ServerConfig.PORT}")
    print(f"API 文件: http://0.0.0.0:{config.ServerConfig.PORT}/docs")
    print(f"網頁測試: 開啟 webrtc_viewer.html")
    print("=" * 60)
    print()
    
    # 啟動 FastAPI server
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=config.ServerConfig.PORT,
        reload=False,
        log_level="info",
    )
