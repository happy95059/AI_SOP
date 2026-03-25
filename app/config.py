"""
配置文件 - TH IPCAM Demo 全局配置
"""
import os
from typing import Tuple, List


# ================================
# RTSP 连接配置
# ================================
class RTSPConfig:
    """RTSP 连接相关配置"""
    
    # 默认 RTSP URL（可通过环境变量 RTSP_URL 覆盖）
    DEFAULT_URL = "rtsp://admin:50984878@192.168.0.205:10554/tcp/av0_0"
    #DEFAULT_URL = "rtsp://192.168.0.109:8554/webcam"
    # RTSP 传输协议选项
    TRANSPORT_OPTIONS = {
        'tcp': 'rtsp_transport;tcp',
        'udp': 'rtsp_transport;udp',
    }
    
    # 连接重试配置
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY_BASE = 2  # 重试延迟基数（秒）
    CONNECTION_TEST_ATTEMPTS = 3  # 连接测试尝试次数
    CONNECTION_TEST_DELAY = 0.5  # 连接测试间隔（秒）
    INITIAL_WAIT = 2  # 初始等待时间（秒）


# ================================
# AI 模型配置
# ================================
class ModelConfig:
    """AI 模型相关配置"""
    
    # 模型文件路径
    #YOLO_SEG_MODEL = "yolov8n-seg.pt"  # YOLO 分割模型
    YOLO_SEG_MODEL = "runs/detect/train11/weights/best.pt"  # YOLO 分割模型
    YOLO_POSE_MODEL = "yolov8n-pose.pt"  # YOLO 姿态估计模型
    
    # 推论设备（None = 自动检测，'cuda' / 'cpu'）
    DEVICE = None
    
    # 模型启用开关
    USE_POSE = False  # 是否启用 YOLOv8-pose 多人骨架
    USE_HANDS = True  # 是否启用 MediaPipe Hands 手部识别
    USE_YOLO = True  # 是否启用 YOLO 物体检测
    
    # MediaPipe Hands 配置
    MAX_HANDS = 4  # 同时检测的手部数量（1-10）
    MIN_DETECTION_CONFIDENCE = 0.5  # 手部检测最小置信度
    MIN_TRACKING_CONFIDENCE = 0.5  # 手部追踪最小置信度
    STATIC_IMAGE_MODE = False  # 是否为静态图像模式
    
    # YOLO 推论配置
    YOLO_VERBOSE = False  # 是否显示 YOLO 详细输出
    YOLO_CONF = 0.35  # YOLO 物体检测置信度阈值
    YOLO_IOU = 0.7  # YOLO NMS IOU 阈值
    POSE_CONF_THRESHOLD = 0.25  # 骨架关键点置信度阈值
    AGNOSTIC_NMS = True  # 是否启用类别无关的 NMS


# ================================
# 绘图配置
# ================================
class DrawConfig:
    """绘图相关配置"""
    
    # 手部附近物体检测
    NEAR_THRESHOLD = 50  # 手部与物体的"附近"距离（像素）
    
    # 颜色配置 (BGR 格式)
    COLOR_HAND_BOX = (0, 255, 0)  # 手部边框颜色（绿色）- 不再使用
    DRAW_HAND_BOX = False  # 是否绘制手部边框
    COLOR_YOLO_BOX = (0, 255, 0)  # YOLO 物体框颜色（绿色）
    COLOR_KEYPOINT = (0, 255, 255)  # 骨架关键点颜色（黄色）
    COLOR_LIMB = (0, 255, 0)  # 骨架连线颜色（绿色）
    COLOR_STATUS_TEXT = (0, 255, 0)  # 状态文字颜色（绿色）
    COLOR_INFO_TEXT = (255, 255, 255)  # 信息文字颜色（白色）
    
    # YOLO 类别颜色映射 (BGR 格式)
    CLASS_COLORS = {
        "semi_finished": (0, 255, 255),  # 半成品 - 黄色
        "finished": (0, 255, 0),  # 成品 - 绿色
        "item": (255, 0, 0),  # 物品 - 蓝色
    }
    
    # 线条和点配置
    BOX_THICKNESS = 2  # 边框线条粗细
    LIMB_THICKNESS = 2  # 骨架连线粗细
    KEYPOINT_RADIUS = 4  # 关键点半径
    
    # 文字配置
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_STATUS = 1.2  # 状态文字大小（放大）
    FONT_SCALE_INFO = 1.0  # 信息文字大小（放大）
    TEXT_THICKNESS = 2  # 文字粗细
    
    # 文字位置
    STATUS_TEXT_POS = (10, 30)  # 状态文字位置
    INFO_TEXT_OFFSET = 10  # 信息文字底部偏移


# ================================
# YOLOv8-pose 骨架配置
# ================================
class PoseConfig:
    """YOLOv8-pose 骨架相关配置"""
    
    # COCO 17 点骨架连线（1-based 索引）
    # 格式: [[起点索引, 终点索引], ...]
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13],  # 头部到颈部
        [12, 13],  # 肩膀连线
        [6, 12], [7, 13],  # 肩膀到躯干
        [6, 7],  # 髋部连线
        [6, 8], [7, 9],  # 髋部到膝盖
        [8, 10], [9, 11],  # 膝盖到脚踝
        [2, 3], [1, 2], [1, 3],  # 脸部
        [2, 4], [3, 5],  # 耳朵到肩膀
        [4, 6], [5, 7],  # 眼睛到肩膀
    ]
    
    # 关键点参考索引（0-based）
    KEYPOINT_NOSE = 0
    KEYPOINT_LEFT_EYE = 1
    KEYPOINT_RIGHT_EYE = 2
    KEYPOINT_LEFT_EAR = 3
    KEYPOINT_RIGHT_EAR = 4
    KEYPOINT_LEFT_SHOULDER = 5
    KEYPOINT_RIGHT_SHOULDER = 6
    KEYPOINT_LEFT_ELBOW = 7
    KEYPOINT_RIGHT_ELBOW = 8
    KEYPOINT_LEFT_WRIST = 9
    KEYPOINT_RIGHT_WRIST = 10
    KEYPOINT_LEFT_HIP = 11
    KEYPOINT_RIGHT_HIP = 12
    KEYPOINT_LEFT_KNEE = 13
    KEYPOINT_RIGHT_KNEE = 14
    KEYPOINT_LEFT_ANKLE = 15
    KEYPOINT_RIGHT_ANKLE = 16


# ================================
# MediaPipe Hands 配置
# ================================
class HandsConfig:
    """MediaPipe Hands 相关配置"""
    
    # 手部关键点参考索引（0-based）
    # 用于"手部附近"物体检测的参考点
    REFERENCE_LANDMARKS = [4, 8, 12]  # 拇指尖(4), 食指尖(8), 中指尖(12)
    
    # 手部边框扩展（像素）
    BOX_PADDING = 10


# ================================
# YOLO 物体检测配置
# ================================
class YOLOConfig:
    """YOLO 物体检测相关配置"""
    
    # 忽略的类别（不绘制边框）
    IGNORED_CLASSES = ["person"]  # 不绘制人形边框
    
    # 类别名称映射（英文 -> 中文）
    CLASS_NAME_MAPPING = {
        "item": "物品",
        "finished": "成品",
        "semi_finished": "半成品",
        "person": "人",
    }


# ================================
# 目标追踪配置
# ================================
class TrackingConfig:
    """目标追踪相关配置（Ultralytics 内建 tracking）"""
    
    # 是否启用目标追踪
    USE_TRACKING = True
    
    # 追踪器类型（botsort.yaml 或 bytetrack.yaml）
    # botsort: 更精确但稍慢
    # bytetrack: 更快但稍微不精确
    TRACKER = "bytetrack.yaml"
    
    # 追踪持久化（保持 tracker 状态）
    PERSIST = True
    
    # 是否显示追踪 ID
    SHOW_TRACK_ID = True
    
    # 追踪 ID 文字配置
    TRACK_ID_COLOR = (255, 255, 0)  # 青色 (BGR)
    TRACK_ID_FONT_SCALE = 0.8
    TRACK_ID_THICKNESS = 2


# ================================
# 区域计数配置
# ================================
class RegionConfig:
    """区域计数相关配置"""
    
    # 是否启用区域框显示
    SHOW_REGION_BOXES = False
    
    # 區域定義 (x1, y1, x2, y2) - 左上角和右下角坐標
    # 左邊區域：半成品區
    REGION_LEFT_X1 = 250
    REGION_LEFT_Y1 = 920
    REGION_LEFT_X2 = 1150
    REGION_LEFT_Y2 = 1250
    
    # 右边区域：成品區
    REGION_RIGHT_X1 = 1300
    REGION_RIGHT_Y1 = 920
    REGION_RIGHT_X2 = 2050
    REGION_RIGHT_Y2 = 1250
    
    # 下方区域：組裝區
    REGION_BOTTOM_X1 = 750
    REGION_BOTTOM_Y1 = 650
    REGION_BOTTOM_X2 = 1650
    REGION_BOTTOM_Y2 = 900
    
    # 區域名稱（中文）
    REGION_LEFT_NAME = "半成品"
    REGION_RIGHT_NAME = "成品"
    REGION_BOTTOM_NAME = "組裝中"
    
    # 區域框顏色 (BGR)
    REGION_BOX_COLOR = (255, 0, 255)  # 紫色
    REGION_BOX_THICKNESS = 4
    
    # 區域計數文字配置
    REGION_TEXT_COLOR = (255, 255, 255)  # 白色
    REGION_TEXT_BG_COLOR = (255, 0, 255)  # 紫色背景
    REGION_TEXT_FONT_SCALE = 1.0  # 放大字體
    REGION_TEXT_THICKNESS = 2
    REGION_TEXT_PADDING = 8  # 文字背景內邊距（增加）


# ================================
# WebRTC 配置
# ================================
class WebRTCConfig:
    """WebRTC 串流相关配置"""
    
    # ICE 服务器配置（内网环境为空）
    ICE_SERVERS = []
    
    # 连接超时配置
    ICE_GATHERING_TIMEOUT = 1.0  # ICE gathering 等待时间（秒）
    CONNECTION_TIMEOUT = 3.0  # 连接失败后清理延迟（秒）


# ================================
# SOP 流程偵測配置
# ================================
class SOPConfig:
    """SOP 組裝流程偵測相關配置"""

    # 是否啟用 SOP 流程偵測
    ENABLED = True

    # ---------- 類別名稱（YOLO 輸出的 class name） ----------
    CLASS_A = "A"                  # 底座
    CLASS_B = "B"                  # 電路板
    CLASS_C = "C"                  # 蓋子
    CLASS_SCREWDRIVER = "screwdriver"  # 螺絲起子

    # ---------- ROI 區域座標 (x1, y1, x2, y2) ----------
    # 組裝區 ROI
    ASSEMBLY_ROI = (1350, 200, 2000, 1200)
    # 輸送帶 ROI（成品放出去的區域）
    CONVEYOR_ROI = (400, 100, 1050, 1200)

    # ---------- 穩定幀數（debounce） ----------
    # 物件在 assembly ROI 內連續偵測到幾幀才算穩定存在
    STABLE_FRAMES_REQUIRED = 5
    # 物件消失幾幀後才真正標記為離開
    MISSING_FRAMES_TOLERANCE = 15

    # ---------- 距離閾值（像素） ----------
    # B 靠近 A 的距離閾值 → 判斷 B 放到 A 上
    DIST_B_TO_A = 120
    # C 靠近 AB 核心位置的距離閾值
    DIST_C_TO_AB = 120
    # 螺絲起子靠近目標物的距離閾值
    DIST_SCREWDRIVER_TO_TARGET = 150
    # 手靠近螺絲起子的距離閾值
    DIST_HAND_TO_SCREWDRIVER = 200

    # ---------- 螺絲起子垂直判定 ----------
    # 螺絲起子 bbox 的 h/w 比例 > 此值視為垂直
    SCREWDRIVER_VERTICAL_RATIO = 1.3

    # ---------- 螺絲鎖附累計幀數 ----------
    # 螺絲起子持續作用幾幀才算鎖附完成
    SCREW_FRAMES_REQUIRED = 10

    # ---------- Rollback ----------
    # Step3 rollback: A 又單獨穩定出現的幀數
    ROLLBACK_STEP3_FRAMES = 8
    # Step5 rollback: B 又單獨穩定出現的幀數
    ROLLBACK_STEP5_FRAMES = 8

    # ---------- Step7: 產品離開 ----------
    # 主體離開 assembly 且進入 conveyor 的持續幀數
    PRODUCT_LEAVE_FRAMES = 8

    # ---------- 歷史紀錄 ----------
    # 最多保留幾件完成品紀錄
    MAX_HISTORY = 50

    # ---------- 主物件選擇策略 ----------
    # 匹配上一幀主物件的最大距離（像素），超過就不認為是同一個
    MAIN_OBJECT_MAX_MATCH_DIST = 200


# ================================
# 服务器配置
# ================================
class ServerConfig:
    """FastAPI 服务器配置"""
    
    # 服务器地址
    HOST = "0.0.0.0"
    PORT = 8000
    
    # CORS 配置
    ALLOW_ORIGINS = ["*"]  # 允许的来源（生产环境应限制）
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]
    
    # Worker 管理配置
    IDLE_TIMEOUT = 180  # Worker 空闲超时时间（秒，3分钟）
    TIMEOUT_CHECK_INTERVAL = 5  # 超时检查间隔（秒）


# ================================
# 日志配置
# ================================
class LogConfig:
    """日志相关配置"""
    
    # 日志级别
    LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # 日志格式
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 特定模块日志级别
    ULTRALYTICS_LEVEL = "ERROR"  # YOLO 库日志级别


# ================================
# 便捷访问函数
# ================================
def get_rtsp_url() -> str:
    """获取 RTSP URL（优先使用环境变量）"""
    return os.environ.get("RTSP_URL", RTSPConfig.DEFAULT_URL)


def get_device() -> str:
    """获取推论设备（自动检测或使用配置）"""
    if ModelConfig.DEVICE is not None:
        return ModelConfig.DEVICE
    
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_opencv_ffmpeg_options(rtsp_url: str = None) -> str:
    """获取 OpenCV FFMPEG 选项"""
    if rtsp_url is None:
        rtsp_url = get_rtsp_url()
    
    if "/tcp/" in rtsp_url:
        return RTSPConfig.TRANSPORT_OPTIONS['tcp']
    else:
        return RTSPConfig.TRANSPORT_OPTIONS['udp']


# ================================
# 配置验证
# ================================
def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 验证 MAX_HANDS 范围
    if not (1 <= ModelConfig.MAX_HANDS <= 10):
        errors.append(f"MAX_HANDS 必须在 1-10 之间，当前值: {ModelConfig.MAX_HANDS}")
    
    # 验证置信度范围
    if not (0.0 <= ModelConfig.MIN_DETECTION_CONFIDENCE <= 1.0):
        errors.append(f"MIN_DETECTION_CONFIDENCE 必须在 0.0-1.0 之间")
    
    if not (0.0 <= ModelConfig.MIN_TRACKING_CONFIDENCE <= 1.0):
        errors.append(f"MIN_TRACKING_CONFIDENCE 必须在 0.0-1.0 之间")
    
    if not (0.0 <= ModelConfig.POSE_CONF_THRESHOLD <= 1.0):
        errors.append(f"POSE_CONF_THRESHOLD 必须在 0.0-1.0 之间")
    
    # 验证阈值
    if DrawConfig.NEAR_THRESHOLD <= 0:
        errors.append(f"NEAR_THRESHOLD 必须大于 0")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    return True


# ================================
# 配置摘要
# ================================
def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("TH IPCAM Demo - 配置摘要")
    print("=" * 60)
    print(f"RTSP URL: {get_rtsp_url()}")
    print(f"推论设备: {get_device()}")
    print(f"启用 Pose: {ModelConfig.USE_POSE}")
    print(f"启用 Hands: {ModelConfig.USE_HANDS}")
    print(f"启用 YOLO: {ModelConfig.USE_YOLO}")
    print(f"最大手部数: {ModelConfig.MAX_HANDS}")
    print(f"手部附近阈值: {DrawConfig.NEAR_THRESHOLD} 像素")
    print(f"服务器地址: {ServerConfig.HOST}:{ServerConfig.PORT}")
    print(f"Worker 超时: {ServerConfig.IDLE_TIMEOUT} 秒")
    print("=" * 60)


if __name__ == "__main__":
    # 验证配置
    validate_config()
    print("✅ 配置验证通过")
    print()
    
    # 打印配置摘要
    print_config_summary()
