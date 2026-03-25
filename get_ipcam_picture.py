"""
IPCAM 拍照/录影工具

模式一（拍照）:
- 显示实时画面
- 按 C 或 SPACE 拍照
- 按 Q 退出
- 图片自动保存到 ipcam_pic 文件夹

模式二（录影）:
- 显示实时画面
- 按 R 开始/停止录影
- 按 Q 退出
- 录影自动保存到 ipcam_video 文件夹
"""
import os
import sys
import time
import argparse
from datetime import datetime

import cv2

# 导入项目模块
from models.RTSPCapture import RTSPCapture

# RTSP 默认设置
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
DEFAULT_RTSP_URL = "rtsp://admin:50984878@192.168.0.205:10554/tcp/av0_0"

# 保存文件的文件夹
SAVE_FOLDER = "ipcam_pic"
VIDEO_FOLDER = "ipcam_video"


def parse_args():
    parser = argparse.ArgumentParser(description="IPCAM 拍照/录影工具")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_RTSP_URL,
        help=f"IPCAM 的 RTSP 网址（默认: {DEFAULT_RTSP_URL}）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["photo", "video"],
        default="photo",
        help="工作模式: photo(拍照) 或 video(录影)，默认: photo",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="保存文件夹（默认: photo模式用ipcam_pic，video模式用ipcam_video）",
    )
    return parser.parse_args()


def create_save_folder(folder_path):
    """创建保存文件夹（如果不存在）"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ 已创建文件夹: {folder_path}")
    else:
        print(f"📁 使用文件夹: {folder_path}")


def get_next_number(folder_path, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mkv']):
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


def draw_photo_overlay(frame, save_folder, photo_count):
    """在画面上绘制拍照模式的操作提示"""
    h, w = frame.shape[:2]
    
    # 半透明背景
    overlay = frame.copy()
    
    # 顶部信息栏
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # 标题
    cv2.putText(
        frame,
        "IPCAM Photo Capture Tool",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    
    # 已拍照数量
    cv2.putText(
        frame,
        f"Photos taken: {photo_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    
    # 底部操作提示
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
    
    cv2.putText(
        frame,
        "Press [C] or [SPACE] to capture photo",
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


def draw_video_overlay(frame, save_folder, is_recording, video_count, recording_duration=0):
    """在画面上绘制录影模式的操作提示"""
    h, w = frame.shape[:2]
    
    # 半透明背景
    overlay = frame.copy()
    
    # 顶部信息栏
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # 标题
    cv2.putText(
        frame,
        "IPCAM Video Recording Tool",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    
    # 录影状态
    if is_recording:
        # 闪烁的 REC 指示器
        if int(time.time() * 2) % 2 == 0:  # 每0.5秒闪烁
            cv2.circle(frame, (w - 50, 30), 15, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (w - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        
        # 录影时长
        minutes = int(recording_duration // 60)
        seconds = int(recording_duration % 60)
        cv2.putText(
            frame,
            f"Recording: {minutes:02d}:{seconds:02d}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            f"Videos recorded: {video_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    
    # 底部操作提示
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
    
    if is_recording:
        cv2.putText(
            frame,
            "Press [R] to STOP recording",
            (10, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Press [R] to START recording",
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


def run_photo_mode(capture, args):
    """运行拍照模式"""
    window_name = "IPCAM Photo Mode - Press C/SPACE to capture, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 获取起始编号
    current_number = get_next_number(args.folder)
    print(f"📝 起始编号: {current_number:03d}")
    print()
    
    photo_count = 0
    last_capture_time = 0
    capture_cooldown = 0.5  # 防止连续拍照，至少间隔 0.5 秒
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                continue
            
            # 复制画面用于显示
            display_frame = frame.copy()
            
            # 绘制信息覆盖层
            draw_photo_overlay(display_frame, args.folder, photo_count)
            
            # 显示画面
            cv2.imshow(window_name, display_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            # 按 C 或 空格键拍照
            if key == ord('c') or key == ord('C') or key == ord(' '):
                if current_time - last_capture_time >= capture_cooldown:
                    # 使用原始画面（不带提示文字）保存
                    if save_image(frame, args.folder, current_number):
                        photo_count += 1
                        current_number += 1
                        last_capture_time = current_time
                        
                        # 简单的视觉反馈：显示一帧白色
                        white_frame = display_frame.copy()
                        white_frame[:] = 255
                        cv2.imshow(window_name, white_frame)
                        cv2.waitKey(50)  # 50ms 闪光效果
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


def run_video_mode(capture, args):
    """运行录影模式"""
    window_name = "IPCAM Video Mode - Press R to start/stop recording, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 获取视频参数
    ret, frame = capture.read()
    if not ret or frame is None:
        print("❌ 无法读取画面", file=sys.stderr)
        return
    
    h, w = frame.shape[:2]
    fps = 20  # 录影帧率
    
    # 获取起始编号
    current_number = get_next_number(args.folder)
    print(f"📝 起始编号: {current_number:03d}")
    print()
    
    video_count = 0
    is_recording = False
    video_writer = None
    recording_start_time = 0
    current_video_path = None
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                continue
            
            # 如果正在录影，写入影片
            if is_recording and video_writer is not None:
                video_writer.write(frame)
            
            # 复制画面用于显示
            display_frame = frame.copy()
            
            # 计算录影时长
            recording_duration = 0
            if is_recording:
                recording_duration = time.time() - recording_start_time
            
            # 绘制信息覆盖层
            draw_video_overlay(display_frame, args.folder, is_recording, video_count, recording_duration)
            
            # 显示画面
            cv2.imshow(window_name, display_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            # 按 R 键开始/停止录影
            if key == ord('r') or key == ord('R'):
                if not is_recording:
                    # 开始录影
                    filename = f"{current_number:03d}.mp4"
                    current_video_path = os.path.join(args.folder, filename)
                    
                    # 创建 VideoWriter (使用 mp4v 编码器)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(current_video_path, fourcc, fps, (w, h))
                    
                    if video_writer.isOpened():
                        is_recording = True
                        recording_start_time = time.time()
                        print(f"🎬 开始录影: {current_video_path}")
                    else:
                        print("❌ 无法创建录影文件", file=sys.stderr)
                        video_writer = None
                else:
                    # 停止录影
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    
                    duration = time.time() - recording_start_time
                    is_recording = False
                    
                    print(f"⏹️  录影结束！已保存: {current_video_path} (时长: {duration:.1f}秒)")
                    
                    video_count += 1
                    current_number += 1
                    current_video_path = None
            
            # 按 Q 退出
            elif key == ord('q') or key == ord('Q'):
                if is_recording:
                    print("\n⚠️  正在录影中，先停止录影...")
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    
                    duration = time.time() - recording_start_time
                    print(f"⏹️  录影已保存: {current_video_path} (时长: {duration:.1f}秒)")
                    video_count += 1
                
                print("\n👋 退出录影工具...")
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出...")
        if is_recording and video_writer is not None:
            video_writer.release()
            duration = time.time() - recording_start_time
            print(f"⏹️  录影已保存: {current_video_path} (时长: {duration:.1f}秒)")
    
    finally:
        if video_writer is not None:
            video_writer.release()
        
        cv2.destroyAllWindows()
        
        print("=" * 60)
        print(f"📊 总共录制: {video_count} 个视频")
        print(f"📁 保存位置: {os.path.abspath(args.folder)}")
        print("=" * 60)


def main():
    args = parse_args()
    
    # 根据模式设置文件夹
    if args.folder is None:
        args.folder = SAVE_FOLDER if args.mode == "photo" else VIDEO_FOLDER
    
    mode_name = "拍照" if args.mode == "photo" else "录影"
    print("=" * 60)
    print(f"IPCAM {mode_name}工具")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"RTSP URL: {args.url}")
    
    # 创建保存文件夹
    create_save_folder(args.folder)
    
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
    
    # 根据模式运行不同的逻辑
    if args.mode == "photo":
        run_photo_mode(capture, args)
    else:
        run_video_mode(capture, args)
    
    # 释放资源
    capture.release()
    print("✅ 已结束。")


if __name__ == "__main__":
    main()
