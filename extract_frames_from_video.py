"""
从视频中提取帧工具

功能：
- 播放视频文件
- 按 SPACE 键截取当前帧并保存
- 按 Q 退出
- 按 ← → 键快进/快退
- 按 P 键暂停/继续
- 图片自动保存到 ipcam_pic 文件夹（按编号命名）
"""
import os
import sys
import argparse

import cv2


# 默认设置
DEFAULT_SAVE_FOLDER = "ipcam_pic/20260320"


def parse_args():
    parser = argparse.ArgumentParser(description="从视频中提取帧工具")
    parser.add_argument(
        "--video",
        type=str,
        help="视频文件路径",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=DEFAULT_SAVE_FOLDER,
        help=f"图片保存文件夹（默认: {DEFAULT_SAVE_FOLDER}）",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=30,
        help="快进/快退的帧数（默认: 30帧）",
    )
    return parser.parse_args()


def create_save_folder(folder_path):
    """创建保存文件夹（如果不存在）"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ 已创建文件夹: {folder_path}")
    else:
        print(f"📁 使用文件夹: {folder_path}")


def get_next_number(folder_path):
    """获取下一个可用的编号"""
    if not os.path.exists(folder_path):
        return 1
    
    max_num = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not file.startswith('.'):
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
        print(f"📸 截取成功！已保存: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}", file=sys.stderr)
        return False


def format_time(seconds):
    """格式化时间为 MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def draw_video_info(frame, current_frame, total_frames, fps, is_paused, frame_count, save_folder):
    """在画面上绘制视频信息"""
    h, w = frame.shape[:2]
    
    # 半透明背景
    overlay = frame.copy()
    
    # 顶部信息栏
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # 标题
    cv2.putText(
        frame,
        "Video Frame Extractor",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    
    # 视频进度信息
    current_time = current_frame / fps if fps > 0 else 0
    total_time = total_frames / fps if fps > 0 else 0
    progress_text = f"Progress: {format_time(current_time)} / {format_time(total_time)} ({current_frame}/{total_frames} frames)"
    cv2.putText(
        frame,
        progress_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    
    # 已提取帧数
    cv2.putText(
        frame,
        f"Frames extracted: {frame_count}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    
    # 暂停指示器
    if is_paused:
        pause_text = "PAUSED"
        (text_width, text_height), _ = cv2.getTextSize(
            pause_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            3
        )
        cv2.putText(
            frame,
            pause_text,
            (w - text_width - 10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
        )
    
    # 进度条
    bar_x = 10
    bar_y = h - 85
    bar_width = w - 20
    bar_height = 15
    
    # 背景（灰色）
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
    
    # 进度（绿色）
    progress = current_frame / total_frames if total_frames > 0 else 0
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
    
    # 边框
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
    
    # 底部操作提示
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
    
    cv2.putText(
        frame,
        "[SPACE]=Capture  [P]=Pause/Play  [←→]=Skip  [Q]=Quit",
        (10, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
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


def main():
    args = parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"❌ 视频文件不存在: {args.video}", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("视频帧提取工具")
    print("=" * 60)
    print(f"视频文件: {args.video}")
    
    # 创建保存文件夹
    create_save_folder(args.folder)
    
    # 打开视频文件
    print(f"\n正在打开视频...")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print("❌ 无法打开视频文件", file=sys.stderr)
        sys.exit(1)
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ 视频已打开")
    print(f"   分辨率: {width}x{height}")
    print(f"   帧率: {fps:.2f} FPS")
    print(f"   总帧数: {total_frames}")
    print(f"   时长: {format_time(total_frames / fps if fps > 0 else 0)}")
    print("\n开始播放...")
    print("=" * 60)
    
    # 创建窗口
    window_name = "Video Frame Extractor - Press SPACE to extract, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 获取起始编号
    current_number = get_next_number(args.folder)
    print(f"📝 起始编号: {current_number:03d}")
    print()
    
    frame_count = 0
    is_paused = False
    current_frame = 0
    
    try:
        while True:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n📽️  视频播放完毕！")
                    break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                # 暂停时保持当前画面
                ret, frame = cap.read()
                if not ret:
                    break
                # 回退一帧以保持当前画面
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            # 复制画面用于显示
            display_frame = frame.copy()
            
            # 绘制信息覆盖层
            draw_video_info(display_frame, current_frame, total_frames, fps, is_paused, frame_count, args.folder)
            
            # 显示画面
            cv2.imshow(window_name, display_frame)
            
            # 按键处理
            wait_time = 1 if is_paused else int(1000 / fps) if fps > 0 else 33
            key = cv2.waitKey(wait_time) & 0xFF
            
            # 按 SPACE 键截取当前帧
            if key == ord(' '):
                # 使用原始画面（不带提示文字）保存
                if save_image(frame, args.folder, current_number):
                    frame_count += 1
                    current_number += 1
                    
                    # 简单的视觉反馈：显示一帧白色
                    white_frame = display_frame.copy()
                    white_frame[:] = 255
                    cv2.imshow(window_name, white_frame)
                    cv2.waitKey(50)  # 50ms 闪光效果
            
            # 按 P 键暂停/继续
            elif key == ord('p') or key == ord('P'):
                is_paused = not is_paused
                if is_paused:
                    print("⏸️  已暂停")
                else:
                    print("▶️  继续播放")
            
            # 按 → 键快进
            elif key == 83 or key == ord('d') or key == ord('D'):  # 右箭头键或 D
                new_frame = min(current_frame + args.skip, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                current_frame = new_frame
                print(f"⏩ 快进到: {format_time(current_frame / fps)} ({current_frame}/{total_frames})")
            
            # 按 ← 键快退
            elif key == 81 or key == ord('a') or key == ord('A'):  # 左箭头键或 A
                new_frame = max(current_frame - args.skip, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                current_frame = new_frame
                print(f"⏪ 快退到: {format_time(current_frame / fps)} ({current_frame}/{total_frames})")
            
            # 按 Q 退出
            elif key == ord('q') or key == ord('Q'):
                print("\n👋 退出视频播放...")
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("=" * 60)
        print(f"📊 总共提取: {frame_count} 帧")
        print(f"📁 保存位置: {os.path.abspath(args.folder)}")
        print("=" * 60)
        print("✅ 已结束。")


if __name__ == "__main__":
    main()
