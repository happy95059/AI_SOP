#!/usr/bin/env python3
"""
IPCAM YOLO 拍照工具

功能：
- 显示带有 YOLO 检测的实时画面
- 按 C 或 SPACE 拍照（保存原始 frame，不带检测框）
- 按 Q 退出
- 可指定 YOLO 模型、保存文件夹、RTSP URL
- 照片自动编号（继续现有最大编号）
"""

import os
import sys
import cv2
import time
import argparse
from datetime import datetime

# 导入项目模块
from models.RTSPCapture import RTSPCapture

# RTSP 默认设置
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
DEFAULT_RTSP_URL = "rtsp://admin:50984878@192.168.0.205:10554/tcp/av0_0"

# 默认设置
DEFAULT_MODEL = "runs/detect/train11/weights/best.pt"
DEFAULT_SAVE_FOLDER = "ipcam_pic/20260324"

# YOLO 检测设置
YOLO_CONF = 0.3  # 置信度阈值
YOLO_IOU = 0.5  # NMS IOU 阈值

# 类别颜色映射 (BGR 格式)
CLASS_COLORS = {
    "semi_finished": (0, 255, 255),  # 半成品 - 黄色
    "finished": (0, 255, 0),  # 成品 - 绿色
    "item": (255, 0, 0),  # 物品 - 蓝色
}

# 类别名称映射（英文 -> 中文）
CLASS_NAME_MAPPING = {
    "item": "物品",
    "finished": "成品",
    "semi_finished": "半成品",
    "person": "人",
}


def parse_args():
    parser = argparse.ArgumentParser(description="IPCAM YOLO 拍照工具")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_RTSP_URL,
        help=f"IPCAM 的 RTSP 网址（默认: {DEFAULT_RTSP_URL}）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"YOLO 模型路径（默认: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=DEFAULT_SAVE_FOLDER,
        help=f"保存文件夹（默认: {DEFAULT_SAVE_FOLDER}）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=YOLO_CONF,
        help=f"YOLO 置信度阈值（默认: {YOLO_CONF}）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=YOLO_IOU,
        help=f"YOLO NMS IOU 阈值（默认: {YOLO_IOU}）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推论设备（cuda/cpu），默认自动检测",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="启用目标追踪（使用 Ultralytics 内建追踪器）",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="追踪器类型（bytetrack.yaml 或 botsort.yaml，默认: bytetrack.yaml）",
    )
    return parser.parse_args()


def create_save_folder(folder_path):
    """创建保存文件夹（如果不存在）"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ 已创建文件夹: {folder_path}")
    else:
        print(f"📁 使用文件夹: {folder_path}")


def get_next_number(folder_path, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """获取下一个可用的编号"""
    if not os.path.exists(folder_path):
        return 1
    
    max_num = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(tuple(extensions)) and not file.startswith('.'):
            # 尝试从文件名中提取数字
            name = os.path.splitext(file)[0]
            if name.isdigit():
                num = int(name)
                max_num = max(max_num, num)
    
    return max_num + 1


def save_image(frame, folder_path, number):
    """保存图片，使用数字编号命名"""
    filename = f"{number:03d}.jpg"
    filepath = os.path.join(folder_path, filename)
    
    try:
        cv2.imwrite(filepath, frame)
        print(f"📸 拍照成功！已保存: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}", file=sys.stderr)
        return False


def get_device(device_arg=None):
    """获取推论设备"""
    if device_arg is not None:
        return device_arg
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 自动检测设备: {device}")
        return device
    except ImportError:
        print(f"🔧 使用设备: cpu")
        return "cpu"


def load_yolo_model(model_path, device):
    """加载 YOLO 模型"""
    try:
        from ultralytics import YOLO
        print(f"📦 正在加载模型: {model_path}")
        model = YOLO(model_path)
        print(f"✅ 模型加载成功")
        return model
    except ImportError:
        print(f"❌ 无法导入 ultralytics，请安装: pip install ultralytics", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}", file=sys.stderr)
        sys.exit(1)


def draw_yolo_detections(frame, results, class_colors=None, class_name_mapping=None, show_track_id=True):
    """在画面上绘制 YOLO 检测结果（支持追踪 ID）"""
    if results is None or len(results) == 0:
        return frame
    
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return frame
    
    names = r.names or {}
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    
    # 获取追踪 ID（如果有的话）
    track_ids = None
    if hasattr(r.boxes, 'id') and r.boxes.id is not None:
        track_ids = r.boxes.id.cpu().numpy().astype(int)
    
    for i in range(len(xyxy)):
        cls_id = int(cls_ids[i])
        cls_name = names.get(cls_id, str(cls_id))
        conf = confs[i]
        
        # 忽略 person 类别
        if cls_name == "person":
            continue
        
        # 获取颜色
        if class_colors and cls_name in class_colors:
            color = class_colors[cls_name]
        else:
            color = (0, 255, 0)
        
        # 获取中文名称
        if class_name_mapping and cls_name in class_name_mapping:
            display_name = class_name_mapping[cls_name]
        else:
            display_name = cls_name
        
        # 绘制边框
        x1, y1, x2, y2 = map(int, xyxy[i])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 构建标签（包含追踪 ID）
        if track_ids is not None and show_track_id:
            track_id = track_ids[i]
            label = f"{display_name} ID:{track_id} {conf:.2f}"
        else:
            label = f"{display_name} {conf:.2f}"
        
        # 绘制标签背景
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )
        
        # 确保标签不会超出画面
        label_y1 = max(y1 - label_height - 10, 0)
        label_y2 = label_y1 + label_height + 10
        
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + label_width + 10, label_y2),
            color,
            -1
        )
        
        # 绘制标签文字
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return frame


def draw_overlay(frame, save_folder, photo_count, model_name, fps=0, tracking_enabled=False):
    """在画面上绘制信息覆盖层"""
    h, w = frame.shape[:2]
    
    # 半透明背景
    overlay = frame.copy()
    
    # 顶部信息栏
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # 标题
    tracking_text = " (Tracking ON)" if tracking_enabled else ""
    cv2.putText(
        frame,
        f"IPCAM YOLO Photo Capture{tracking_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    
    # 模型信息
    model_short = os.path.basename(model_name)
    cv2.putText(
        frame,
        f"Model: {model_short}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    
    # 已拍照数量和 FPS
    cv2.putText(
        frame,
        f"Photos: {photo_count} | FPS: {fps:.1f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    
    # 底部操作提示
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
    
    cv2.putText(
        frame,
        "Press [C] or [SPACE] to capture (saves original frame)",
        (10, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    
    cv2.putText(
        frame,
        "Press [Q] to quit",
        (10, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    
    # 保存路径提示（右下角）
    folder_text = f"Save to: {save_folder}"
    (text_width, text_height), _ = cv2.getTextSize(
        folder_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )
    cv2.putText(
        frame,
        folder_text,
        (w - text_width - 10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )


def run_yolo_photo_mode(capture, model, args):
    """运行 YOLO 拍照模式"""
    window_name = "YOLO Photo Mode - Press C/SPACE to capture, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 获取起始编号
    current_number = get_next_number(args.folder)
    print(f"📝 起始编号: {current_number:03d}")
    print()
    
    photo_count = 0
    last_capture_time = 0
    capture_cooldown = 0.5  # 防止连续拍照，至少间隔 0.5 秒
    
    # FPS 计算
    fps = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            # 读取原始画面
            ret, frame = capture.read()
            if not ret or frame is None:
                continue
            
            # 保存原始画面（用于拍照）
            original_frame = frame.copy()
            
            # YOLO 检测或追踪
            try:
                if args.track:
                    # 使用追踪模式
                    results = model.track(
                        frame,
                        conf=args.conf,
                        iou=args.iou,
                        verbose=False,
                        device=args.device,
                        persist=True,
                        tracker=args.tracker,
                    )
                else:
                    # 使用普通检测模式
                    results = model(
                        frame,
                        conf=args.conf,
                        iou=args.iou,
                        verbose=False,
                        device=args.device
                    )
                
                # 在画面上绘制检测结果
                display_frame = frame.copy()
                display_frame = draw_yolo_detections(
                    display_frame,
                    results,
                    class_colors=CLASS_COLORS,
                    class_name_mapping=CLASS_NAME_MAPPING,
                    show_track_id=args.track
                )
            except Exception as e:
                print(f"⚠️  YOLO 检测错误: {e}")
                display_frame = frame.copy()
            
            # 计算 FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # 绘制信息覆盖层
            draw_overlay(display_frame, args.folder, photo_count, args.model, fps, tracking_enabled=args.track)
            
            # 显示画面
            cv2.imshow(window_name, display_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            # 按 C 或 空格键拍照
            if key == ord('c') or key == ord('C') or key == ord(' '):
                if current_time - last_capture_time >= capture_cooldown:
                    # 使用原始画面（不带 YOLO 检测框）保存
                    if save_image(original_frame, args.folder, current_number):
                        photo_count += 1
                        current_number += 1
                        last_capture_time = current_time
                else:
                    print("⚠️  拍照太快了，请稍等...")
            
            # 按 Q 退出
            elif key == ord('q') or key == ord('Q'):
                print("\n👋 退出拍照工具...")
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出...")
    
    finally:
        cv2.destroyAllWindows()
        
        print("=" * 60)
        print(f"📊 总共拍摄: {photo_count} 张照片")
        print(f"📁 保存位置: {os.path.abspath(args.folder)}")
        print("=" * 60)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("IPCAM YOLO 拍照工具")
    print("=" * 60)
    print(f"RTSP URL: {args.url}")
    print(f"YOLO 模型: {args.model}")
    print(f"保存文件夹: {args.folder}")
    print(f"置信度阈值: {args.conf}")
    print(f"NMS IOU 阈值: {args.iou}")
    print(f"目标追踪: {'启用 (' + args.tracker + ')' if args.track else '未启用'}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    # 创建保存文件夹
    create_save_folder(args.folder)
    
    # 获取推论设备
    args.device = get_device(args.device)
    
    # 加载 YOLO 模型
    model = load_yolo_model(args.model, args.device)
    
    # 连接 IPCAM
    print(f"\n正在连接 IPCAM...")
    capture = RTSPCapture(args.url)
    time.sleep(2)  # 等待连接建立
    
    # 测试连接
    ret, frame = capture.read()
    if not ret or frame is None:
        print("❌ 无法连接到 IPCAM，请检查 RTSP URL 和网络连接。", file=sys.stderr)
        capture.release()
        sys.exit(1)
    
    h, w = frame.shape[:2]
    print(f"✅ 连接成功！分辨率: {w}x{h}")
    print("\n开始显示画面...")
    print("=" * 60)
    
    # 运行 YOLO 拍照模式
    run_yolo_photo_mode(capture, model, args)
    
    # 释放资源
    capture.release()
    print("✅ 已结束。")


if __name__ == "__main__":
    main()
