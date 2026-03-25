"""
MainObjectSelector - 主物件選擇器

從多個 A/B/C 偵測中挑出 assembly ROI 內最可能是主流程的 mainA/mainB/mainC。
不做全畫面完整多目標追蹤，只追 assembly ROI 內的主實例。
"""
import math
import logging
from typing import Optional, Tuple, List, Dict

from ..config import SOPConfig

logger = logging.getLogger(__name__)


class MainObjectSelector:
    """
    從每幀的 YOLO detections 中，為每個類別 (A/B/C/screwdriver) 挑出主實例。

    選擇策略：
      1. 只考慮 assembly ROI 內的偵測（外面不重要）
      2. 優先靠近上一幀 main object 的位置
      3. 沒有上一幀時，選 ROI 內面積最大的
    """

    def __init__(self):
        self._assembly_roi = SOPConfig.ASSEMBLY_ROI
        self._roi_center = (
            (self._assembly_roi[0] + self._assembly_roi[2]) // 2,
            (self._assembly_roi[1] + self._assembly_roi[3]) // 2,
        )
        self._max_match_dist = SOPConfig.MAIN_OBJECT_MAX_MATCH_DIST

    def select(
        self,
        detections_by_class: Dict[str, List[Tuple[int, int, int, int]]],
        prev_centers: Dict[str, Optional[Tuple[int, int]]],
    ) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        """
        為每個類別選出主實例 bbox。

        Args:
            detections_by_class: {class_name: [bbox, bbox, ...]}
                class_name 為 SOPConfig 中定義的 A/B/C/screwdriver
                bbox 為 (x1, y1, x2, y2)
            prev_centers: {class_name: (cx, cy) or None}
                上一幀主物件的中心點

        Returns:
            {class_name: selected_bbox or None}
        """
        result: Dict[str, Optional[Tuple[int, int, int, int]]] = {}

        for cls_name in [SOPConfig.CLASS_A, SOPConfig.CLASS_B,
                         SOPConfig.CLASS_C, SOPConfig.CLASS_SCREWDRIVER]:
            bboxes = detections_by_class.get(cls_name, [])
            prev_center = prev_centers.get(cls_name)
            result[cls_name] = self._pick_best(bboxes, prev_center)

        return result

    # ------------------------------------------------------------------

    def _pick_best(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        prev_center: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if not bboxes:
            return None

        # 只考慮 assembly ROI 內的偵測，外面的世界不重要
        in_roi = [b for b in bboxes if self._in_roi(self._bbox_center(b))]
        if not in_roi:
            return None

        # 如果有上一幀位置，優先靠近上一幀
        if prev_center is not None:
            best = min(in_roi, key=lambda b: self._dist(self._bbox_center(b), prev_center))
            if self._dist(self._bbox_center(best), prev_center) <= self._max_match_dist:
                return best

        # 沒有上一幀 or 距離太遠 → ROI 內面積最大
        return max(in_roi, key=self._bbox_area)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    @staticmethod
    def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _in_roi(self, point: Tuple[int, int]) -> bool:
        x, y = point
        r = self._assembly_roi
        return r[0] <= x <= r[2] and r[1] <= y <= r[3]
