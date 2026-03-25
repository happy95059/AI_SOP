"""
VideoWorker - 影像處理工作執行緒
負責 IPCam 連線、影像讀取、YOLO/MediaPipe 推論、框圖繪製
"""
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from models.RTSPCapture import RTSPCapture
from .config import (
    ModelConfig,
    DrawConfig,
    PoseConfig,
    HandsConfig,
    YOLOConfig,
    TrackingConfig,
    RegionConfig,
    RTSPConfig,
    SOPConfig,
    get_device,
)
from .sop import SOPSystem

# YOLO 模組
try:
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

logger = logging.getLogger(__name__)


class VideoWorker:
    """影像處理工作執行緒，產生 latest_frame"""
    
    def __init__(
        self,
        rtsp_url: str,
        use_pose: bool = None,
        use_hands: bool = None,
        use_yolo: bool = None,
        use_tracking: bool = None,
        device: Optional[str] = None,
        near_threshold: int = None,
        max_hands: int = None,
    ):
        """
        初始化 VideoWorker
        
        Args:
            rtsp_url: RTSP 網址
            use_pose: 是否啟用 YOLOv8-pose 多人骨架（None = 使用配置文件默認值）
            use_hands: 是否啟用 MediaPipe Hands 手部辨識（None = 使用配置文件默認值）
            use_yolo: 是否啟用 YOLO 物體偵測（None = 使用配置文件默認值）
            use_tracking: 是否啟用目標追踪（None = 使用配置文件默認值）
            device: 推論裝置（cuda/cpu），None 則自動選擇
            near_threshold: 手部附近物體閾值（像素，None = 使用配置文件默認值）
            max_hands: 同時偵測的手部數量（None = 使用配置文件默認值）
        """
        self.rtsp_url = rtsp_url
        
        # 使用配置文件的默認值（如果參數為 None）
        self.use_pose = use_pose if use_pose is not None else ModelConfig.USE_POSE
        self.use_hands = use_hands if use_hands is not None else ModelConfig.USE_HANDS
        self.use_yolo = (use_yolo if use_yolo is not None else ModelConfig.USE_YOLO) and HAS_YOLO
        self.use_tracking = (use_tracking if use_tracking is not None else TrackingConfig.USE_TRACKING) and HAS_YOLO
        self.near_threshold = near_threshold if near_threshold is not None else DrawConfig.NEAR_THRESHOLD
        self.max_hands = max_hands if max_hands is not None else ModelConfig.MAX_HANDS
        
        # 推論裝置
        if device is None:
            self.device = get_device()
        else:
            self.device = device.strip().lower()
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("未偵測到 CUDA，改為使用 CPU")
                self.device = "cpu"
        
        logger.info(f"使用裝置: {self.device}")
        
        # 執行緒控制
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # 最新影格
        self._latest_frame: Optional[np.ndarray] = None
        
        # RTSP 擷取器
        self._capture: Optional[RTSPCapture] = None
        
        # MediaPipe Hands
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._hands = None
        
        # YOLO 模型
        self._yolo_model = None
        self._pose_model = None
        
        # SOP 流程偵測系統
        self._sop_system: Optional[SOPSystem] = None
        if SOPConfig.ENABLED:
            self._sop_system = SOPSystem()
        
        # SOP 事件佇列（供 WebSocket 消費）
        self._sop_events: list = []
        self._sop_events_lock = threading.Lock()
        
        # 狀態
        self._is_running = False
        self._error: Optional[str] = None
        
        # 中文字體快取（使用 PIL）
        self._chinese_font_cache = {}
        self._load_chinese_font()
    
    def _load_chinese_font(self):
        """載入中文字體（macOS 系統字體）"""
        import platform
        system = platform.system()
        
        # macOS 字體路徑
        if system == "Darwin":
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        # Linux 字體路徑
        elif system == "Linux":
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
            ]
        # Windows 字體路徑
        else:
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",  # 微軟雅黑
                "C:/Windows/Fonts/simsun.ttc",  # 宋體
            ]
        
        # 嘗試載入字體
        for font_path in font_paths:
            try:
                # 預載入常用字體大小
                for size in [20, 30, 40, 50, 60, 80]:
                    self._chinese_font_cache[size] = ImageFont.truetype(font_path, size)
                logger.info(f"成功載入中文字體: {font_path}")
                return
            except Exception as e:
                continue
        
        # 如果都失敗，使用默認字體
        logger.warning("無法載入中文字體，使用默認字體（可能無法顯示中文）")
        for size in [20, 30, 40, 50, 60, 80]:
            self._chinese_font_cache[size] = ImageFont.load_default()
    
    def _put_chinese_text(self, frame, text, position, font_size=30, color=(255, 255, 255), bg_color=None, padding=5):
        """在 frame 上繪製中文文字（使用 PIL）
        
        Args:
            frame: OpenCV 圖像 (BGR)
            text: 要顯示的文字
            position: (x, y) 文字左上角位置
            font_size: 字體大小
            color: 文字顏色 (BGR)
            bg_color: 背景顏色 (BGR)，None 表示無背景
            padding: 背景內邊距
        
        Returns:
            修改後的 frame
        """
        # 找最接近的字體大小
        available_sizes = sorted(self._chinese_font_cache.keys())
        closest_size = min(available_sizes, key=lambda x: abs(x - font_size))
        font = self._chinese_font_cache[closest_size]
        
        # 轉換為 PIL Image (BGR -> RGB)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 計算文字邊界框
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = position
        
        # 繪製背景（如果指定）
        if bg_color is not None:
            bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])  # BGR -> RGB
            draw.rectangle(
                [(x - padding, y - padding), 
                 (x + text_width + padding, y + text_height + padding)],
                fill=bg_color_rgb
            )
        
        # 繪製文字 (BGR -> RGB)
        color_rgb = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=color_rgb)
        
        # 轉回 OpenCV 格式 (RGB -> BGR)
        frame_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return frame_with_text
    
    def start(self):
        """啟動工作執行緒"""
        if self._is_running:
            logger.warning("VideoWorker 已在執行中")
            return
        
        logger.info("啟動 VideoWorker...")
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self):
        """停止工作執行緒"""
        if not self._is_running:
            return
        
        logger.info("停止 VideoWorker...")
        self._stop_event.set()
        
        # 等待執行緒結束
        if self._thread and self._thread.is_alive():
            logger.info("等待 VideoWorker 執行緒結束...")
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("VideoWorker 執行緒未在 5 秒內結束")
            self._thread = None
        
        # 清理資源
        self._cleanup()
        self._is_running = False
        logger.info("VideoWorker 已停止")
    
    def _cleanup(self):
        """清理資源"""
        logger.info("清理 VideoWorker 資源...")
        
        # 清理 RTSP capture
        if self._capture:
            try:
                logger.info("釋放 RTSP capture...")
                self._capture.release()
                logger.info("RTSP capture 已釋放")
            except Exception as e:
                logger.error(f"釋放 RTSP capture 失敗: {e}")
            finally:
                self._capture = None
        
        # 清理 MediaPipe Hands
        if self._hands:
            try:
                logger.info("關閉 MediaPipe Hands...")
                self._hands.close()
                logger.info("MediaPipe Hands 已關閉")
            except Exception as e:
                logger.error(f"關閉 MediaPipe Hands 失敗: {e}")
            finally:
                self._hands = None
        
        # 清理 YOLO 模型（釋放記憶體）
        self._yolo_model = None
        self._pose_model = None
        
        logger.info("VideoWorker 資源清理完成")
    
    def _run(self):
        """主要執行緒邏輯"""
        try:
            self._is_running = True
            self._initialize()
            self._process_loop()
        except Exception as e:
            logger.exception(f"VideoWorker 執行錯誤: {e}")
            self._error = str(e)
        finally:
            self._cleanup()
            self._is_running = False
    
    def _initialize(self):
        """初始化模型與連線"""
        logger.info("初始化模型...")
        
        # 載入 YOLOv8-pose
        if self.use_pose and HAS_YOLO:
            try:
                logger.info(f"載入 YOLOv8-pose 模型（{ModelConfig.YOLO_POSE_MODEL}）...")
                self._pose_model = YOLO(ModelConfig.YOLO_POSE_MODEL)
                logger.info("YOLOv8-pose 已啟用")
            except Exception as e:
                logger.error(f"YOLOv8-pose 載入失敗: {e}")
                self.use_pose = False
        
        # 載入 YOLOv8-seg
        if self.use_yolo:
            try:
                logger.info(f"載入 YOLOv8-seg 模型（{ModelConfig.YOLO_SEG_MODEL}）...")
                self._yolo_model = YOLO(ModelConfig.YOLO_SEG_MODEL)
                logger.info("YOLOv8-seg 已啟用")
            except Exception as e:
                logger.error(f"YOLOv8-seg 載入失敗: {e}")
                self.use_yolo = False
        
        # 初始化 MediaPipe Hands
        if self.use_hands:
            try:
                logger.info("初始化 MediaPipe Hands...")
                self._hands = self._mp_hands.Hands(
                    static_image_mode=ModelConfig.STATIC_IMAGE_MODE,
                    max_num_hands=max(1, min(self.max_hands, 10)),
                    min_detection_confidence=ModelConfig.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=ModelConfig.MIN_TRACKING_CONFIDENCE,
                )
                logger.info("MediaPipe Hands 已初始化")
            except Exception as e:
                logger.error(f"MediaPipe Hands 初始化失敗: {e}")
                self.use_hands = False
                self._hands = None
        
        # 連接 RTSP（帶重試機制）
        logger.info(f"連接 RTSP: {self.rtsp_url}")
        max_retries = RTSPConfig.MAX_RETRIES
        for attempt in range(max_retries):
            try:
                self._capture = RTSPCapture(self.rtsp_url)
                time.sleep(RTSPConfig.INITIAL_WAIT)  # 等待連線建立
                
                # 測試連線
                test_success = False
                for test_attempt in range(RTSPConfig.CONNECTION_TEST_ATTEMPTS):
                    ret, frame = self._capture.read()
                    if ret and frame is not None:
                        test_success = True
                        logger.info(f"RTSP 連線成功，解析度: {frame.shape[1]}x{frame.shape[0]}")
                        break
                    time.sleep(RTSPConfig.CONNECTION_TEST_DELAY)
                
                if test_success:
                    break
                else:
                    raise RuntimeError("無法取得影格，RTSP 連線測試失敗")
                    
            except Exception as e:
                logger.error(f"RTSP 連線嘗試 {attempt + 1}/{max_retries} 失敗: {e}")
                if self._capture:
                    try:
                        self._capture.release()
                    except:
                        pass
                    self._capture = None
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * RTSPConfig.RETRY_DELAY_BASE
                    logger.info(f"等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"RTSP 連線失敗（已重試 {max_retries} 次）: {e}")
    
    def _process_loop(self):
        """影像處理迴圈"""
        logger.info("開始影像處理迴圈...")
        
        # 統計資訊
        frame_count = 0
        failed_read_count = 0
        last_log_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                # 讀取影格
                ret, frame = self._capture.read()
                if not ret or frame is None:
                    failed_read_count += 1
                    continue
                
                # 處理影格
                processed_frame = self._process_frame(frame)
                frame_count += 1
                
                # 更新 latest_frame
                with self._lock:
                    self._latest_frame = processed_frame
                
                # 每 5 秒記錄一次統計資訊
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    fps = frame_count / (current_time - last_log_time)
                    logger.info(f"處理統計 - FPS: {fps:.2f}, 已處理: {frame_count} 幀, 讀取失敗: {failed_read_count} 次")
                    frame_count = 0
                    failed_read_count = 0
                    last_log_time = current_time
                
            except Exception as e:
                logger.error(f"處理影格錯誤: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """處理單一影格：推論 + 繪圖"""
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        # 統計資訊（用於調試）
        detections_info = {
            'hands': 0,
            'pose': 0,
            'yolo_objects': 0,
        }
        
        # MediaPipe Hands
        results_hands_list = None
        if self.use_hands and self._hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self._hands.process(rgb)
            results_hands_list = results_hands.multi_hand_landmarks
            if results_hands_list:
                detections_info['hands'] = len(results_hands_list)
        
        # YOLOv8-pose：多人骨架（已關閉）
        # if self.use_pose and self._pose_model is not None:
        #     try:
        #         results_pose_yolo = self._pose_model(
        #             frame,
        #             verbose=ModelConfig.YOLO_VERBOSE,
        #             device=self.device
        #         )
        #         if results_pose_yolo and len(results_pose_yolo) > 0 and results_pose_yolo[0].keypoints is not None:
        #             detections_info['pose'] = len(results_pose_yolo[0].keypoints.data)
        #             self._draw_pose_yolo_multiple(
        #                 display_frame,
        #                 results_pose_yolo[0].keypoints.data,
        #                 conf_thres=ModelConfig.POSE_CONF_THRESHOLD,
        #                 kpt_color=DrawConfig.COLOR_KEYPOINT,
        #                 limb_color=DrawConfig.COLOR_LIMB,
        #                 radius=DrawConfig.KEYPOINT_RADIUS,
        #                 thickness=DrawConfig.LIMB_THICKNESS,
        #             )
        #     except Exception as e:
        #         logger.debug(f"Pose 推論錯誤: {e}")
        
        # YOLO 分割：執行推論
        results_yolo = None
        if self.use_yolo and self._yolo_model is not None:
            try:
                # 如果啟用追踪，使用 track() 方法
                if self.use_tracking:
                    results_yolo = self._yolo_model.track(
                        frame,
                        verbose=ModelConfig.YOLO_VERBOSE,
                        conf=ModelConfig.YOLO_CONF,
                        iou=ModelConfig.YOLO_IOU,
                        device=self.device,
                        agnostic_nms=ModelConfig.AGNOSTIC_NMS,
                        persist=TrackingConfig.PERSIST,
                        tracker=TrackingConfig.TRACKER,
                    )
                else:
                    # 不使用追踪，正常检测
                    results_yolo = self._yolo_model(
                        frame,
                        verbose=ModelConfig.YOLO_VERBOSE,
                        conf=ModelConfig.YOLO_CONF,
                        iou=ModelConfig.YOLO_IOU,
                        device=self.device,
                        agnostic_nms=ModelConfig.AGNOSTIC_NMS,
                    )
                
                # 統計檢測到的物體數量
                if results_yolo and len(results_yolo) > 0:
                    r = results_yolo[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        detections_info['yolo_objects'] = len(r.boxes)
                
                # # 只畫手部附近物體的矩形框（如果啟用手部檢測）
                # if self.use_hands and results_hands_list:
                #     hand_points = self._get_hand_points(frame, results_hands_list)
                #     if hand_points:
                #         self._draw_yolo_near_boxes(
                #             display_frame,
                #             results_yolo,
                #             hand_points,
                #             near_threshold=self.near_threshold,
                #             box_color=DrawConfig.COLOR_YOLO_BOX,
                #             thickness=DrawConfig.BOX_THICKNESS,
                #         )
                # # 如果未啟用手部檢測，直接畫所有物體框
                # else:
                #     self._draw_yolo_seg(
                #         display_frame,
                #         results_yolo,
                #         box_color=DrawConfig.COLOR_YOLO_BOX,
                #         thickness=DrawConfig.BOX_THICKNESS,
                #     )
                self._draw_yolo_seg(
                        display_frame,
                        results_yolo,
                        box_color=DrawConfig.COLOR_YOLO_BOX,
                        thickness=DrawConfig.BOX_THICKNESS,
                )
                
            except Exception as e:
                logger.debug(f"YOLO 推論錯誤: {e}")
        
        # 計數並繪製區域（無論 YOLO 是否啟用，區域框都會顯示）
        region_counts = self._count_and_draw_regions(display_frame, results_yolo)
        
        # 在畫面中間上方大大顯示計數統計
        if region_counts:
            self._draw_center_top_summary(display_frame, region_counts)
        
        # SOP 流程偵測
        if self._sop_system is not None:
            hand_centers, hand_bboxes = self._extract_hand_data(frame, results_hands_list)
            sop_state, sop_events = self._sop_system.process_frame(
                results_yolo, hand_centers, hand_bboxes
            )
            # 儲存事件供外部消費
            if sop_events:
                with self._sop_events_lock:
                    self._sop_events.extend(sop_events)
            # 在畫面上顯示 SOP 狀態
            self._draw_sop_overlay(display_frame, sop_state)
        
        # 繪製手部
        if self.use_hands and results_hands_list:
            self._draw_hands(display_frame, results_hands_list)
        
        # 繪製狀態資訊
        status_text = f"Pose: {self.use_pose} | Hands: {self.use_hands} | YOLO: {self.use_yolo} | Track: {self.use_tracking}"
        cv2.putText(
            display_frame,
            status_text,
            DrawConfig.STATUS_TEXT_POS,
            DrawConfig.FONT,
            DrawConfig.FONT_SCALE_STATUS,
            DrawConfig.COLOR_STATUS_TEXT,
            DrawConfig.TEXT_THICKNESS,
        )
        
        # 繪製檢測統計資訊（用於調試）
        stats_text = f"Detections - Hands:{detections_info['hands']} Pose:{detections_info['pose']} Objects:{detections_info['yolo_objects']}"
        cv2.putText(
            display_frame,
            stats_text,
            (DrawConfig.STATUS_TEXT_POS[0], DrawConfig.STATUS_TEXT_POS[1] + 50),
            DrawConfig.FONT,
            DrawConfig.FONT_SCALE_INFO,
            DrawConfig.COLOR_INFO_TEXT,
            DrawConfig.TEXT_THICKNESS,
        )
        
        return display_frame
    
    def _get_hand_points(self, frame, results_hands_list):
        """從 MediaPipe 手部結果取得參考點"""
        if results_hands_list is None:
            return []
        h, w = frame.shape[:2]
        points = []
        for hand_landmarks in results_hands_list:
            if hand_landmarks is None:
                continue
            # 使用配置文件中定義的參考點
            for idx in HandsConfig.REFERENCE_LANDMARKS:
                lm = hand_landmarks.landmark[idx]
                points.append((int(lm.x * w), int(lm.y * h)))
        return points
    
    def _draw_yolo_near_boxes(self, frame, results_yolo, hand_points, near_threshold=50, box_color=(0, 255, 0), thickness=2):
        """只畫出「手部附近」的 YOLO 物體矩形框；忽略配置中指定的類別"""
        if results_yolo is None or len(results_yolo) == 0:
            return
        r = results_yolo[0]
        if r.boxes is None or len(r.boxes) == 0:
            return
        names = r.names or {}
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        for i in range(len(xyxy)):
            cls_id = int(cls_ids[i])
            # 檢查是否為忽略類別
            if names.get(cls_id, "") in YOLOConfig.IGNORED_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, xyxy[i])
            is_near_hand = False
            for hp in hand_points:
                px, py = hp[0], hp[1]
                if x1 - near_threshold <= px <= x2 + near_threshold and y1 - near_threshold <= py <= y2 + near_threshold:
                    is_near_hand = True
                    break
            if is_near_hand:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
    
    def _draw_yolo_seg(self, frame, results_yolo, box_color=(0, 255, 0), thickness=2):
        """畫出所有 YOLO 物體矩形框（根據類別使用不同顏色）；忽略配置中指定的類別；如果啟用追踪則顯示 ID"""
        if results_yolo is None or len(results_yolo) == 0:
            return
        r = results_yolo[0]
        if r.boxes is None or len(r.boxes) == 0:
            return
        names = r.names or {}
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        
        # 獲取追踪 ID（如果有的話）
        track_ids = None
        if hasattr(r.boxes, 'id') and r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)
        
        # 類別顏色映射 (BGR 格式)
        CLASS_COLORS = {
            "semi_finished": (0, 255, 255),  # 半成品 - 黄色
            "finished": (0, 255, 0),  # 成品 - 绿色
            "item": (255, 0, 0),  # 物品 - 蓝色
        }
        
        for i in range(len(xyxy)):
            cls_id = int(cls_ids[i])
            # 檢查是否為忽略類別
            if names.get(cls_id, "") in YOLOConfig.IGNORED_CLASSES:
                continue
            
            # 獲取類別名稱
            class_name = names.get(cls_id, f"class_{cls_id}")
            
            # 根據類別選擇顏色
            current_box_color = CLASS_COLORS.get(class_name, box_color)
            
            x1, y1, x2, y2 = map(int, xyxy[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), current_box_color, thickness)
            
            # 顯示類別名稱（轉換成中文）
            class_name_cn = YOLOConfig.CLASS_NAME_MAPPING.get(class_name, class_name)
            
            # 如果啟用追踪且有 ID，添加 ID 到標籤
            if self.use_tracking and track_ids is not None and TrackingConfig.SHOW_TRACK_ID:
                track_id = track_ids[i]
                label = f"{class_name_cn} ID:{track_id}"
            else:
                label = f"{class_name_cn}"
            
            # 使用 PIL 繪製中文標籤
            frame[:] = self._put_chinese_text(
                frame, 
                label, 
                (x1 + 5, y1 - 25),
                font_size=20,
                color=(0, 0, 0),
                bg_color=current_box_color,
                padding=5
            )
    
    def _count_and_draw_regions(self, frame, results_yolo):
        """計數並繪製區域框和物體數量（即使沒有檢測結果也會顯示區域框）
        
        Returns:
            dict: 各區域的計數資訊 {區域名稱: 數量}
        """
        # 準備檢測結果數據
        xyxy = None
        cls_ids = None
        names = {}
        
        if results_yolo is not None and len(results_yolo) > 0:
            r = results_yolo[0]
            if r.boxes is not None and len(r.boxes) > 0:
                names = r.names or {}
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()
        
        # 定義三個區域
        regions = [
            {
                "name": RegionConfig.REGION_LEFT_NAME,
                "x1": RegionConfig.REGION_LEFT_X1,
                "y1": RegionConfig.REGION_LEFT_Y1,
                "x2": RegionConfig.REGION_LEFT_X2,
                "y2": RegionConfig.REGION_LEFT_Y2,
                "count": 0
            },
            {
                "name": RegionConfig.REGION_RIGHT_NAME,
                "x1": RegionConfig.REGION_RIGHT_X1,
                "y1": RegionConfig.REGION_RIGHT_Y1,
                "x2": RegionConfig.REGION_RIGHT_X2,
                "y2": RegionConfig.REGION_RIGHT_Y2,
                "count": 0
            },
            {
                "name": RegionConfig.REGION_BOTTOM_NAME,
                "x1": RegionConfig.REGION_BOTTOM_X1,
                "y1": RegionConfig.REGION_BOTTOM_Y1,
                "x2": RegionConfig.REGION_BOTTOM_X2,
                "y2": RegionConfig.REGION_BOTTOM_Y2,
                "count": 0
            },
        ]
        
        # 計數每個區域內的物體（使用物體中心點判斷）
        if xyxy is not None and cls_ids is not None:
            for i in range(len(xyxy)):
                cls_id = int(cls_ids[i])
                # 忽略指定類別
                if names.get(cls_id, "") in YOLOConfig.IGNORED_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, xyxy[i])
                # 計算物體中心點
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # 檢查物體是否在各個區域內
                for region in regions:
                    if (region["x1"] <= cx <= region["x2"] and 
                        region["y1"] <= cy <= region["y2"]):
                        # 組裝中區域只記錄 0/1（有無物體），不計數
                        if region["name"] == "組裝中":
                            region["count"] = 1
                        else:
                            region["count"] += 1
        
        # 繪製區域框（如果啟用）
        if RegionConfig.SHOW_REGION_BOXES:
            for region in regions:
                cv2.rectangle(
                    frame,
                    (region["x1"], region["y1"]),
                    (region["x2"], region["y2"]),
                    RegionConfig.REGION_BOX_COLOR,
                    RegionConfig.REGION_BOX_THICKNESS
                )
        
        # 不再繪製區域計數文字（已移至畫面上方統一顯示）
        
        # 返回計數資訊供右上角顯示使用
        return {region['name']: region['count'] for region in regions}
    
    def _draw_center_top_summary(self, frame, region_counts):
        """在畫面中間上方大大顯示半成品、成品和組裝中的狀態"""
        h, w = frame.shape[:2]
        
        # 獲取三個區域的計數
        semi_count = region_counts.get('半成品', 0)
        finished_count = region_counts.get('成品', 0)
        assembly_status = region_counts.get('組裝中', 0)
        
        # 準備文字
        text_semi = f"半成品: {semi_count}"
        text_finished = f"成品: {finished_count}"
        text_assembly = f"組裝中: {'是' if assembly_status == 1 else '否'}"
        
        # 大字體配置
        font_size = 60
        padding = 30
        
        # 繪製半透明黑色背景（簡化版，固定大小）
        bg_width = 1200
        bg_height = 120
        bg_x1 = (w - bg_width) // 2
        bg_y1 = 20
        bg_x2 = bg_x1 + bg_width
        bg_y2 = bg_y1 + bg_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 使用 PIL 繪製中文文字
        text_y = bg_y1 + 30
        
        # 半成品 (黃色)
        text_x_semi = bg_x1 + padding
        frame[:] = self._put_chinese_text(
            frame,
            text_semi,
            (text_x_semi, text_y),
            font_size=font_size,
            color=(0, 255, 255),
            bg_color=None
        )
        
        # 組裝中 (紅色)
        text_x_assembly = bg_x1 + 350
        frame[:] = self._put_chinese_text(
            frame,
            text_assembly,
            (text_x_assembly, text_y),
            font_size=font_size,
            color=(0, 0, 255),
            bg_color=None
        )
        
        # 成品 (綠色)
        text_x_finished = bg_x1 + 700
        frame[:] = self._put_chinese_text(
            frame,
            text_finished,
            (text_x_finished, text_y),
            font_size=font_size,
            color=(0, 255, 0),
            bg_color=None
        )
    
    def _draw_pose_yolo_multiple(self, frame, keypoints_data, conf_thres=0.25, kpt_color=(0, 255, 255), limb_color=(0, 255, 0), radius=4, thickness=2):
        """將 YOLOv8-pose 多人 keypoints 畫成骨架到 frame 上（多人）"""
        if keypoints_data is None:
            return
        try:
            kpts = keypoints_data.cpu().numpy()
        except Exception:
            kpts = np.asarray(keypoints_data)
        if kpts.ndim == 2:
            kpts = kpts[None, :, :]
        h, w = frame.shape[:2]
        for p in range(kpts.shape[0]):
            k = kpts[p]  # (17, 2) or (17, 3)
            if k.shape[0] < 17:
                continue
            has_conf = k.shape[1] >= 3
            # 使用配置文件中的骨架定義
            for i, sk in enumerate(PoseConfig.SKELETON):
                i1, i2 = sk[0] - 1, sk[1] - 1
                if i1 >= k.shape[0] or i2 >= k.shape[0]:
                    continue
                if has_conf and k.shape[1] >= 3:
                    if k[i1, 2] < conf_thres or k[i2, 2] < conf_thres:
                        continue
                x1, y1 = int(k[i1, 0]), int(k[i1, 1])
                x2, y2 = int(k[i2, 0]), int(k[i2, 1])
                if x1 <= 0 and y1 <= 0 and x2 <= 0 and y2 <= 0:
                    continue
                cv2.line(frame, (x1, y1), (x2, y2), limb_color, thickness, lineType=cv2.LINE_AA)
            for i in range(min(17, k.shape[0])):
                if has_conf and k.shape[1] >= 3 and k[i, 2] < conf_thres:
                    continue
                x, y = int(k[i, 0]), int(k[i, 1])
                if x <= 0 and y <= 0:
                    continue
                cv2.circle(frame, (x, y), radius, kpt_color, -1, lineType=cv2.LINE_AA)
    
    def _draw_hands(self, frame, results_hands_list):
        """將 MediaPipe Hands 結果繪製到 frame 上（只繪製骨架，不繪製框框）"""
        if results_hands_list is None:
            return
        h, w = frame.shape[:2]
        for hand_landmarks in results_hands_list:
            if hand_landmarks is None:
                continue
            
            # 不再繪製手部邊框（已關閉）
            # xs = [lm.x * w for lm in hand_landmarks.landmark]
            # ys = [lm.y * h for lm in hand_landmarks.landmark]
            # padding = HandsConfig.BOX_PADDING
            # x_min = max(0, int(min(xs)) - padding)
            # y_min = max(0, int(min(ys)) - padding)
            # x_max = min(w, int(max(xs)) + padding)
            # y_max = min(h, int(max(ys)) + padding)
            # cv2.rectangle(
            #     frame,
            #     (x_min, y_min),
            #     (x_max, y_max),
            #     DrawConfig.COLOR_HAND_BOX,
            #     DrawConfig.BOX_THICKNESS
            # )
            
            # 繪製手部骨架（保留）
            self._mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )
    
    # ==================================================================
    # SOP 輔助方法
    # ==================================================================

    def _extract_hand_data(self, frame, results_hands_list):
        """從 MediaPipe 手部結果取得所有手的中心點和 bbox"""
        if results_hands_list is None:
            return [], []
        h, w = frame.shape[:2]
        centers = []
        bboxes = []
        for hand_landmarks in results_hands_list:
            if hand_landmarks is None:
                continue
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            centers.append((cx, cy))
            bboxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        return centers, bboxes

    def _draw_sop_overlay(self, frame, sop_state: dict):
        """在畫面上疊加 SOP 流程資訊"""
        h, w = frame.shape[:2]

        step = sop_state.get('current_step', 0)
        debug_msg = sop_state.get('debug_msg', '')

        # SOP 步驟定義
        step_names = {
            -1: '等待上工',
            0: '等待開始',
            1: '放入底座(A)',
            2: '放入電路板(B)',
            3: 'B放到A上',
            4: '第一段鎖螺絲',
            5: '放上蓋子(C)',
            6: '第二段鎖螺絲',
            7: '放到輸送帶',
        }
        step_label = step_names.get(step, f'Step {step}')

        # 畫半透明背景
        bg_y1 = h - 160
        bg_y2 = h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bg_y1), (w, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 大字顯示當前步驟
        sop_text = f"SOP Step {step}: {step_label}"
        frame[:] = self._put_chinese_text(
            frame, sop_text,
            (20, bg_y1 + 15),
            font_size=50,
            color=(0, 255, 255),
        )

        # 小字顯示 debug 資訊
        if debug_msg:
            frame[:] = self._put_chinese_text(
                frame, debug_msg,
                (20, bg_y1 + 80),
                font_size=30,
                color=(200, 200, 200),
            )

        # 繪製 assembly ROI 框
        a_roi = SOPConfig.ASSEMBLY_ROI
        cv2.rectangle(frame, (a_roi[0], a_roi[1]), (a_roi[2], a_roi[3]),
                      (0, 255, 0), 2)
        frame[:] = self._put_chinese_text(
            frame, '組裝區',
            (a_roi[0] + 5, a_roi[1] - 30),
            font_size=20, color=(0, 255, 0),
        )

        # 繪製 conveyor ROI 框
        c_roi = SOPConfig.CONVEYOR_ROI
        cv2.rectangle(frame, (c_roi[0], c_roi[1]), (c_roi[2], c_roi[3]),
                      (255, 165, 0), 2)
        frame[:] = self._put_chinese_text(
            frame, '輸送帶',
            (c_roi[0] + 5, c_roi[1] - 30),
            font_size=20, color=(255, 165, 0),
        )

    # ==================================================================
    # SOP 對外介面
    # ==================================================================

    def get_sop_state(self) -> Optional[dict]:
        """取得當前 SOP 狀態（執行緒安全）"""
        if self._sop_system is None:
            return None
        return self._sop_system.get_state_dict()

    def drain_sop_events(self) -> list:
        """取出並清空所有累積的 SOP 事件（執行緒安全）"""
        with self._sop_events_lock:
            events = self._sop_events
            self._sop_events = []
            return events

    def get_sop_history(self) -> list:
        """取得 SOP 歷史完成品紀錄"""
        if self._sop_system is None:
            return []
        return self._sop_system.get_history()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """取得最新的處理後影格（執行緒安全）"""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None
    
    @property
    def is_running(self) -> bool:
        """檢查是否正在執行"""
        return self._is_running
    
    @property
    def error(self) -> Optional[str]:
        """取得錯誤訊息（如果有）"""
        return self._error
