"""
產線工作站影像識別主程式
- 使用 RTSPCapture 抓取 IPCAM 影像
- 使用 MediaPipe 辨識骨架（Pose）與手指位置（Hands）
- 使用 FastSAM 辨識手部框內物品（可選 --no-sam 關閉）
"""
import argparse
import os
import sys
import time

import cv2
import torch

# 專案模組
from models.RTSPCapture import RTSPCapture

# MediaPipe：骨架與手部
import mediapipe as mp

# FastSAM：手部框內物品分割（可選）
# 抑制 Ultralytics「settings reset to default」提示（不影響執行）
try:
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
except Exception:
    pass
try:
    from ultralytics import FastSAM
    HAS_FASTSAM = True
except ImportError:
    HAS_FASTSAM = False

# ===============================
# RTSP 設定（可在此修改預設連線）
# ===============================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
RTSP_URL = "rtsp://admin:50984878@192.168.0.206:10554/udp/av0_0"

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser(description="產線工作站影像識別：骨架與手指辨識")
    parser.add_argument(
        "--url",
        type=str,
        default=RTSP_URL,
        help=f"IPCAM 的 RTSP 網址（預設使用 main.py 內 RTSP_URL）",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="關閉骨架（Pose）辨識",
    )
    parser.add_argument(
        "--no-hands",
        action="store_true",
        help="關閉手指（Hands）辨識",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不顯示視窗（僅擷取與辨識，適用無螢幕環境）",
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="關閉 FastSAM 手部框內物品分割",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推論裝置：cuda / cuda:0 / cpu（預設：有 GPU 則用 cuda）",
    )
    return parser.parse_args()


def draw_sam_contours(frame, result, contour_color=(0, 255, 255), thickness=2):
    """將 FastSAM 結果的每個 mask 畫成輪廓到 frame 上。"""
    if result is None or result.masks is None:
        return
    masks = result.masks.data
    if masks is None or masks.numel() == 0:
        return
    try:
        masks_np = masks.cpu().numpy()
    except Exception:
        return
    h, w = frame.shape[:2]
    for i in range(masks_np.shape[0]):
        mask = (masks_np[i] * 255).astype("uint8")
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, contour_color, thickness)


def get_hand_boxes(frame, results_hands_list):
    """從 MediaPipe 手部結果取得每隻手的邊界框 [x_min, y_min, x_max, y_max] 列表"""
    if results_hands_list is None:
        return []
    h, w = frame.shape[:2]
    boxes = []
    for hand_landmarks in results_hands_list:
        if hand_landmarks is None:
            continue
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        x_min = max(0, int(min(xs)) - 10)
        y_min = max(0, int(min(ys)) - 10)
        x_max = min(w, int(max(xs)) + 10)
        y_max = min(h, int(max(ys)) + 10)
        if x_max > x_min and y_max > y_min:
            boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def draw_pose(frame, results_pose, mp_pose, mp_drawing, mp_drawing_styles):
    """將 MediaPipe Pose 結果繪製到 frame 上，並框出人體範圍"""
    if results_pose.pose_landmarks is None:
        return
    h, w = frame.shape[:2]
    lm = results_pose.pose_landmarks.landmark
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    x_min = max(0, int(min(xs)) - 15)
    y_min = max(0, int(min(ys)) - 15)
    x_max = min(w, int(max(xs)) + 15)
    y_max = min(h, int(max(ys)) + 15)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    mp_drawing.draw_landmarks(
        frame,
        results_pose.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )


def draw_hands(frame, results_hands_list, mp_hands, mp_drawing, mp_drawing_styles):
    """將 MediaPipe Hands 結果繪製到 frame 上，並框出手部範圍"""
    if results_hands_list is None:
        return
    h, w = frame.shape[:2]
    for hand_landmarks in results_hands_list:
        if hand_landmarks is None:
            continue
        # 依關鍵點計算手部邊界框
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        x_min = max(0, int(min(xs)) - 10)
        y_min = max(0, int(min(ys)) - 10)
        x_max = min(w, int(max(xs)) + 10)
        y_max = min(h, int(max(ys)) + 10)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


def main():
    args = parse_args()

    # MediaPipe 初始化
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    use_pose = not args.no_pose
    use_hands = not args.no_hands
    use_sam = not args.no_sam and HAS_FASTSAM
    if args.no_sam and not HAS_FASTSAM:
        print("未安裝 ultralytics，FastSAM 已關閉。", file=sys.stderr)
    sam_model = None
    # GPU：預設有 CUDA 則用 cuda，可用 --device 指定 cuda / cuda:0 / cpu
    if args.device is not None:
        device = args.device.strip().lower()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("未偵測到 CUDA，改為使用 CPU。", file=sys.stderr)
        device = "cpu"
    print(f"使用裝置: {device}" + (" (GPU)" if device.startswith("cuda") else " (CPU)"))
    if use_sam:
        print("載入 FastSAM 模型（FastSAM-s.pt 首次會自動下載）…")
        try:
            sam_model = FastSAM("FastSAM-s.pt")
            print("FastSAM 已啟用，裝置:", device)
        except Exception as e:
            print("FastSAM 載入失敗，已關閉 SAM:", e, file=sys.stderr)
            use_sam = False

    # RTSP 擷取
    print(f"連接 RTSP: {args.url}")
    capture = RTSPCapture(args.url)

    # 給 RTSP 幾秒建立連線
    time.sleep(2)

    ret, frame = capture.read()
    if not ret or frame is None:
        print("無法取得影格，請檢查 RTSP URL 與網路。", file=sys.stderr)
        capture.release()
        sys.exit(1)

    h, w = frame.shape[:2]

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        print("開始辨識（按 Q 結束）。Pose:", use_pose, "Hands:", use_hands)

        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                continue

            # 不左右反轉，顯示攝影機原始畫面
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_pose = None
            results_hands_list = None

            if use_pose:
                results_pose = pose.process(rgb)
            if use_hands:
                results_hands = hands.process(rgb)
                results_hands_list = results_hands.multi_hand_landmarks

            # 手部框內 FastSAM 分割（辨識框內物品並畫出輪廓）
            display_frame = frame
            if use_sam and sam_model is not None and use_hands and results_hands_list:
                hand_boxes = get_hand_boxes(frame, results_hands_list)
                if hand_boxes:
                    try:
                        results_sam = sam_model.predict(
                            frame,
                            bboxes=hand_boxes,
                            device=device,
                            conf=0.4,
                            iou=0.9,
                            retina_masks=True,
                        )
                        res = (
                            results_sam[0]
                            if (results_sam and isinstance(results_sam, (list, tuple)))
                            else (results_sam if results_sam else None)
                        )
                        if res is not None and len(res) > 0:
                            draw_sam_contours(
                                display_frame,
                                res,
                                contour_color=(0, 255, 255),
                                thickness=2,
                            )
                    except Exception:
                        pass

            # 繪製結果（人體框、手部框與關鍵點）
            if use_pose and results_pose:
                draw_pose(
                    display_frame, results_pose, mp_pose, mp_drawing, mp_drawing_styles
                )
            if use_hands and results_hands_list:
                draw_hands(
                    display_frame,
                    results_hands_list,
                    mp_hands,
                    mp_drawing,
                    mp_drawing_styles,
                )

            # 顯示狀態列
            cv2.putText(
                display_frame,
                "Pose: ON | Hands: ON" if (use_pose and use_hands) else f"Pose: {use_pose} | Hands: {use_hands}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Press Q to quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if not args.no_display:
                cv2.imshow("TH_IPCAM_DEMO - Pose & Hands & SAM", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if not args.no_display:
            cv2.destroyAllWindows()

    capture.release()
    print("已結束。")


if __name__ == "__main__":
    main()
