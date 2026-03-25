# TH_IPCAM_DEMO

產線工作站影像識別專案：透過 IPCAM 即時辨識人員動作、手指位置與骨架，供後續安全或作業流程分析使用。

---

## 專案目標

- **人員動作識別**：辨識產線工作站中人員的動作狀態
- **手指位置**：偵測手指關鍵點與位置
- **骨架偵測**：以人體骨架關鍵點描繪姿態

---

## 架構概覽

```
main.py              ← 主程式入口（影像擷取 + MediaPipe 辨識流程）
models/
  RTSPCapture.py     ← IPCAM RTSP 影像擷取模組
```

1. **使用 `RTSPCapture.py` 抓取 IPCAM 影像**  
   透過 RTSP 串流取得網路攝影機畫面，以後端執行緒持續讀取最新影格，降低延遲。

2. **使用 MediaPipe 辨識骨架與手指**  
   以 MediaPipe Pose 辨識人體骨架、以 MediaPipe Hands 辨識手指關鍵點，即時繪製於畫面上。

3. **使用 FastSAM 辨識手部框內物品**  
   以手部邊界框為 prompt，對框內區域做 FastSAM 分割，標示手部框內偵測到的物品（可加 `--no-sam` 關閉以減輕負載）。

---

## 模組說明

### `models/RTSPCapture.py`

- **功能**：以 OpenCV（FFmpeg 後端）連接 RTSP URL，在背景執行緒持續讀取影格。
- **介面**：
  - `__init__(url)`：傳入 RTSP 網址，例如 `rtsp://user:pass@ip:554/stream`
  - `read()`：回傳 `(ret, frame)`，與 `cv2.VideoCapture.read()` 用法一致
  - `release()`：停止執行緒並釋放資源
- **依賴**：`opencv-python`（需具備 FFmpeg 支援）、`threading`（標準庫）。

### `main.py`

- 負責：
  1. 建立 `RTSPCapture` 並連接 IPCAM
  2. 迴圈讀取影格
  3. 使用 MediaPipe Pose 辨識骨架、MediaPipe Hands 辨識手指
  4. 使用 FastSAM 對手部框內區域做物品分割（需安裝 ultralytics，可加 `--no-sam` 關閉）
  5. 將關鍵點與分割結果繪製於畫面上並顯示（可選 `--no-display` 不開視窗）
- 命令列參數：`--url`、`--no-pose`、`--no-hands`、`--no-display`、`--no-sam`、`--device`（GPU/CPU）

---

## 環境需求

- **Python**：建議 3.8+
- **必要套件**：見專案根目錄 `requirements.txt`
  - `opencv-python`：影像讀取與 RTSP
  - `mediapipe`：骨架（Pose）與手指（Hands）辨識
  - `ultralytics`、`torch`、`torchvision`：FastSAM 手部框內物品分割（可選）
  - `numpy`、`protobuf`、`absl-py` 等 MediaPipe 依賴

安裝：

```bash
pip install -r requirements.txt
```

- 系統需支援 **FFmpeg**（OpenCV 透過 FFmpeg 解 RTSP）。
- **GPU**：若有 NVIDIA GPU 與 CUDA，FastSAM 會自動使用 GPU；可用 `--device cuda` 或 `--device cuda:0` 指定，或 `--device cpu` 強制用 CPU。

---

## RTSP 設定

在 `main.py` 頂部有 RTSP 設定區塊，可修改預設連線：

- **OPENCV_FFMPEG_CAPTURE_OPTIONS**：設為 `rtsp_transport;tcp` 可提高 RTSP 穩定性。
- **RTSP_URL**：預設 IPCAM 的 RTSP 網址（含帳密若需要）。  
  執行時未加 `--url` 即使用此預設值；命令列 `--url` 可覆蓋。

---

## 使用方式

1. 在 `main.py` 內修改 **RTSP_URL**，或執行時以 `--url` 指定 IPCAM 的 RTSP 網址。
2. 在專案根目錄執行：

```bash
# 使用 main.py 內 RTSP 設定（RTSP_URL）
python main.py

# 以命令列指定 RTSP 網址（覆蓋預設）
python main.py --url "rtsp://user:pass@192.168.1.100:554/stream"

# 使用 GPU（預設有 CUDA 即用 GPU）
python main.py

# 指定裝置：GPU 第一張卡 / 強制 CPU
python main.py --device cuda:0
python main.py --device cpu

# 僅骨架、僅手指、關閉 SAM、或不顯示視窗
python main.py --url "rtsp://..." --no-hands
python main.py --url "rtsp://..." --no-pose
python main.py --url "rtsp://..." --no-sam
python main.py --url "rtsp://..." --no-display
```

3. 畫面上會即時顯示骨架與手指關鍵點；按 **Q** 結束程式。

---

## 專案規範（.cursorrules 摘要）

- 請勿修改 `File/`、`Tools/`、`Config/` 下受保護的檔案。
- 邏輯擴充請寫在 `main.py` 或指定擴展檔案中。
- 註解不得任意刪改；新增註解需與既有風格一致。
- 除錯時請依「觀察 Log → 最小化變動 → 驗證」進行。
- 本說明檔置於專案根目錄；若程式架構或流程有變更，請同步更新 README.md。

---

## 授權與備註

依專案所屬單位規定使用。  
MediaPipe 使用內建模型，無需額外下載。FastSAM 首次執行會自動下載 `FastSAM-s.pt`。



python convert_coco_to_yolov8.py \
  --coco-json yolov8_001.coco/train/_annotations.coco.json \
  --output-dir train_yolo/data/min001 \
  --val-ratio 0.2

  yolo detect train model=yolov8n.pt data=train_yolo/data/min001/data.yaml epochs=100 imgsz=640

  yolo detect train \
  model=yolov8n.pt \
  data=train_yolo/data/min001/data.yaml \
  epochs=200 \
  imgsz=640 \
  batch=4 \
  patience=30 \
  close_mosaic=0 \
  degrees=5 \
  translate=0.05 \
  scale=0.2 \
  fliplr=0.5

  # 凍結
  yolo detect train \
  model=yolov8n.pt \
  data=train_yolo/data/min001/data.yaml \
  epochs=200 \
  imgsz=640 \
  batch=4 \
  patience=30 \
  freeze=10