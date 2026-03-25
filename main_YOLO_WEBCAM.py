"""
產線工作站影像識別（YOLO 版 - Webcam）
- 使用 WebcamCapture 抓取本機攝影機影像
- 使用 MediaPipe 辨識手指位置（Hands）
- 使用 YOLOv8-pose 顯示多人骨架（可選 --pose）
- 使用 YOLOv8-seg 辨識手部附近的物體，矩形框
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# 專案模組
from models.WebcamCapture import WebcamCapture

# MediaPipe：骨架與手部
import mediapipe as mp

# YOLO 分割：手部附近物體輪廓
try:
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
except Exception:
    pass
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser(description="產線工作站影像識別（YOLO + Webcam）")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam 攝影機編號（0 為預設）",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="顯示多人骨架（YOLOv8-pose）；預設不顯示人形",
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
        "--no-yolo",
        action="store_true",
        help="關閉 YOLO 手部附近物體輪廓",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推論裝置：cuda / cuda:0 / cpu（預設：有 GPU 則用 cuda）",
    )
    parser.add_argument(
        "--near",
        type=int,
        default=50,
        help="手部與物體輪廓的「附近」距離（像素），預設 50",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=4,
        help="同時偵測的手部數量（多人時可調高，預設 4＝約 2 人）",
    )
    return parser.parse_args()


def get_hand_points(frame, results_hands_list):
    """從 MediaPipe 手部結果取得參考點 (食指、拇指等) 的像素座標 [(x,y), ...]"""
    if results_hands_list is None:
        return []
    h, w = frame.shape[:2]
    points = []
    for hand_landmarks in results_hands_list:
        if hand_landmarks is None:
            continue
        # 食指尖(8)、拇指尖(4)、中指尖(12) 作為「手部附近」參考點
        for idx in [4, 8, 12]:
            lm = hand_landmarks.landmark[idx]
            points.append((int(lm.x * w), int(lm.y * h)))
    return points


def draw_yolo_near_boxes(frame, results_yolo, hand_points, near_threshold=50, box_color=(0, 255, 0), thickness=2):
    """只畫出「手部附近」的 YOLO 物體矩形框；人形（person）不畫。"""
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
        if names.get(cls_id, "") == "person":
            continue
        x1, y1, x2, y2 = map(int, xyxy[i])
        is_near_hand = False
        for hp in hand_points:
            # 手部點在框內或距框邊 within 閾值 算「附近」
            px, py = hp[0], hp[1]
            if x1 - near_threshold <= px <= x2 + near_threshold and y1 - near_threshold <= py <= y2 + near_threshold:
                is_near_hand = True
                break
        if is_near_hand:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)


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


# YOLOv8-pose 骨架連線（COCO 17 點，1-based 索引）
POSE_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
]


def draw_pose_yolo_multiple(frame, keypoints_data, conf_thres=0.25, kpt_color=(0, 255, 255), limb_color=(0, 255, 0), radius=4, thickness=2):
    """將 YOLOv8-pose 多人 keypoints 畫成骨架到 frame 上（多人）。"""
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
        for i, sk in enumerate(POSE_SKELETON):
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


def draw_hands(frame, results_hands_list, mp_hands, mp_drawing, mp_drawing_styles):
    """將 MediaPipe Hands 結果繪製到 frame 上，並框出手部範圍"""
    if results_hands_list is None:
        return
    h, w = frame.shape[:2]
    for hand_landmarks in results_hands_list:
        if hand_landmarks is None:
            continue
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

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    use_pose = args.pose
    use_hands = not args.no_hands
    use_yolo = not args.no_yolo and HAS_YOLO
    if args.no_yolo or not HAS_YOLO:
        if not HAS_YOLO:
            print("未安裝 ultralytics，YOLO 手部附近輪廓已關閉。", file=sys.stderr)
    yolo_model = None
    pose_model = None

    if args.device is not None:
        device = args.device.strip().lower()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("未偵測到 CUDA，改為使用 CPU。", file=sys.stderr)
        device = "cpu"
    print(f"使用裝置: {device}" + (" (GPU)" if device.startswith("cuda") else " (CPU)"))

    if use_pose and HAS_YOLO:
        print("載入 YOLOv8-pose 模型（yolov8n-pose.pt 首次會自動下載）…")
        try:
            pose_model = YOLO("yolov8n-pose.pt")
            print("多人骨架（YOLOv8-pose）已啟用，裝置:", device)
        except Exception as e:
            print("YOLOv8-pose 載入失敗，已關閉:", e, file=sys.stderr)
            use_pose = False

    if use_yolo:
        print("載入 YOLOv8-seg 模型（yolov8n-seg.pt 首次會自動下載）…")
        try:
            yolo_model = YOLO("yolov8n-seg.pt")
            print("YOLO 手部附近物體輪廓已啟用，裝置:", device)
        except Exception as e:
            print("YOLO 載入失敗，已關閉:", e, file=sys.stderr)
            use_yolo = False

    print(f"開啟 Webcam 攝影機: {args.camera}")
    capture = WebcamCapture(args.camera)
    time.sleep(1)

    ret, frame = capture.read()
    if not ret or frame is None:
        print("無法取得影格，請檢查攝影機編號與連線。", file=sys.stderr)
        capture.release()
        sys.exit(1)

    h, w = frame.shape[:2]

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max(1, min(args.max_hands, 10)),  # 多人：1～10 隻手
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        print("開始辨識（按 Q 結束）。Pose 多人:", use_pose, "Hands:", use_hands, "YOLO 手部附近:", use_yolo)

        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands_list = None

            if use_hands:
                results_hands = hands.process(rgb)
                results_hands_list = results_hands.multi_hand_landmarks

            display_frame = frame

            # YOLOv8-pose：多人骨架（只畫骨架與關鍵點，不畫人體框）
            if use_pose and pose_model is not None:
                try:
                    results_pose_yolo = pose_model(frame, verbose=False, device=device)
                    if results_pose_yolo and len(results_pose_yolo) > 0 and results_pose_yolo[0].keypoints is not None:
                        draw_pose_yolo_multiple(
                            display_frame,
                            results_pose_yolo[0].keypoints.data,
                            conf_thres=0.25,
                            kpt_color=(0, 255, 255),
                            limb_color=(0, 255, 0),
                            radius=4,
                            thickness=2,
                        )
                except Exception:
                    pass

            # YOLO 分割：只畫手部附近物體的矩形框
            if use_yolo and yolo_model is not None and use_hands and results_hands_list:
                hand_points = get_hand_points(frame, results_hands_list)
                if hand_points:
                    try:
                        results_yolo = yolo_model(frame, verbose=False, device=device)
                        draw_yolo_near_boxes(
                            display_frame,
                            results_yolo,
                            hand_points,
                            near_threshold=args.near,
                            box_color=(0, 255, 0),
                            thickness=2,
                        )
                    except Exception:
                        pass

            if use_hands and results_hands_list:
                draw_hands(
                    display_frame,
                    results_hands_list,
                    mp_hands,
                    mp_drawing,
                    mp_drawing_styles,
                )

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
                cv2.imshow("TH_IPCAM_WEBCAM - Pose & Hands & YOLO", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if not args.no_display:
            cv2.destroyAllWindows()

    capture.release()
    print("已結束。")


if __name__ == "__main__":
    main()
