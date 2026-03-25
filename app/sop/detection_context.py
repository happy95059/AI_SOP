"""
DetectionContext - 每幀整理好的偵測上下文
讓 rule engine 不用直接處理原始 YOLO/MediaPipe 輸出
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class DetectionContext:
    """
    每幀整理好的偵測上下文。

    由 SOPSystem 在每幀建立，供 SOPRuleEngine 使用。
    """

    frame_idx: int = 0

    # --- 主物件（assembly ROI 內選出的） ---
    mainA_bbox: Optional[Tuple[int, int, int, int]] = None
    mainA_center: Optional[Tuple[int, int]] = None
    mainB_bbox: Optional[Tuple[int, int, int, int]] = None
    mainB_center: Optional[Tuple[int, int]] = None
    mainC_bbox: Optional[Tuple[int, int, int, int]] = None
    mainC_center: Optional[Tuple[int, int]] = None

    # --- 螺絲起子 / 手部 ---
    screwdriver_bbox: Optional[Tuple[int, int, int, int]] = None
    screwdriver_center: Optional[Tuple[int, int]] = None
    screwdriver_vertical: bool = False  # bbox h > w * ratio
    hand_centers: List[Tuple[int, int]] = field(default_factory=list)

    # --- ROI flags ---
    A_in_assembly: bool = False
    B_in_assembly: bool = False
    C_in_assembly: bool = False
    screwdriver_in_assembly: bool = False
    product_in_conveyor: bool = False  # 主體離開 assembly 且進入 conveyor

    # --- 預計算距離（-1 表示不可計算） ---
    dist_B_to_A: float = -1.0
    dist_C_to_AB: float = -1.0
    dist_screwdriver_to_target: float = -1.0   # 根據步驟動態填入
    dist_hand_to_screwdriver: float = -1.0

    @staticmethod
    def _dist(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> float:
        if p1 is None or p2 is None:
            return -1.0
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _center_of_bbox(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _point_in_roi(point: Optional[Tuple[int, int]],
                      roi: Tuple[int, int, int, int]) -> bool:
        if point is None:
            return False
        x, y = point
        return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]

    @staticmethod
    def _is_vertical(bbox: Optional[Tuple[int, int, int, int]], ratio: float) -> bool:
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        return (h / w) >= ratio

    @staticmethod
    def _min_dist_to_list(
        point: Optional[Tuple[int, int]],
        points: List[Tuple[int, int]],
    ) -> float:
        """計算 point 到 points 列表中最近的距離"""
        if point is None or not points:
            return -1.0
        return min(math.hypot(point[0] - p[0], point[1] - p[1]) for p in points)
