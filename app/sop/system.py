"""
SOPSystem - SOP 流程偵測總控

職責：
  1. 每幀接收 YOLO / MediaPipe detections
  2. 用 MainObjectSelector 選出 mainA/mainB/mainC
  3. 組裝 DetectionContext
  4. 丟給 SOPRuleEngine.update()
  5. 回傳最新 state + events
  6. Step7 完成後 reset 開始下一件
"""
import logging
import math
from collections import deque
from typing import Optional, Tuple, List, Dict

from ..config import SOPConfig
from .state import SOPState
from .detection_context import DetectionContext
from .selector import MainObjectSelector
from .rule_engine import SOPRuleEngine

logger = logging.getLogger(__name__)

CFG = SOPConfig


class SOPSystem:
    """SOP 流程偵測總控"""

    def __init__(self):
        self.current_state = SOPState()
        self.history: deque = deque(maxlen=CFG.MAX_HISTORY)
        self.selector = MainObjectSelector()
        self.rule_engine = SOPRuleEngine()

        self._frame_idx: int = 0

    # ==================================================================
    # 主要入口
    # ==================================================================

    def process_frame(
        self,
        yolo_results,
        hand_centers: Optional[List[Tuple[int, int]]] = None,
        hand_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[dict, List[dict]]:
        """
        每幀呼叫一次。

        Args:
            yolo_results: ultralytics YOLO 推論結果（results list）
            hand_centers: MediaPipe 手部中心點列表 [(x, y), ...]

        Returns:
            (state_dict, events)
            state_dict: 當前 state 的 dict（供前端即時顯示）
            events: 事件列表（step_changed / product_completed）
        """
        self._frame_idx += 1

        # 1. 解析 YOLO 結果 → detections_by_class
        detections_by_class = self._parse_yolo(yolo_results)

        # 2. 取得上一幀主物件中心
        prev_centers = {
            CFG.CLASS_A: self.current_state.mainA.center,
            CFG.CLASS_B: self.current_state.mainB.center,
            CFG.CLASS_C: self.current_state.mainC.center,
            CFG.CLASS_SCREWDRIVER: None,  # screwdriver 不需要追蹤上一幀
        }

        # 3. 用 selector 選出主實例
        selected = self.selector.select(detections_by_class, prev_centers)

        # 4. 建立 DetectionContext
        ctx = self._build_context(
            selected, hand_centers or [], hand_bboxes or [],
            detections_by_class,
        )

        # 5. 丟給 rule engine
        events = self.rule_engine.update(self.current_state, ctx)

        # 6. 如果產品完成，歸檔 + reset
        if self.current_state.product_done:
            summary = self.current_state.summary()
            self.history.append(summary)
            logger.info(f"[SOP] 歸檔產品 {summary['product_id']}，"
                        f"歷史紀錄數: {len(self.history)}")
            self.current_state = SOPState()

        return self.current_state.to_dict(), events

    # ==================================================================
    # 解析 YOLO 結果
    # ==================================================================

    def _parse_yolo(
        self,
        yolo_results,
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        將 YOLO results 轉成 {class_name: [bbox, ...]}。
        只保留 A/B/C/screwdriver 類別。
        """
        result: Dict[str, List[Tuple[int, int, int, int]]] = {
            CFG.CLASS_A: [],
            CFG.CLASS_B: [],
            CFG.CLASS_C: [],
            CFG.CLASS_SCREWDRIVER: [],
        }

        if yolo_results is None or len(yolo_results) == 0:
            return result

        r = yolo_results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return result

        names = r.names or {}
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            cls_name = names.get(int(cls_ids[i]), "")
            if cls_name in result:
                x1, y1, x2, y2 = map(int, xyxy[i])
                result[cls_name].append((x1, y1, x2, y2))

        return result

    # ==================================================================
    # 建立 DetectionContext
    # ==================================================================

    def _build_context(
        self,
        selected: Dict[str, Optional[Tuple[int, int, int, int]]],
        hand_centers: List[Tuple[int, int]],
        hand_bboxes: List[Tuple[int, int, int, int]],
        detections_by_class: Dict[str, List[Tuple[int, int, int, int]]],
    ) -> DetectionContext:
        ctx = DetectionContext()
        ctx.frame_idx = self._frame_idx

        # --- 主物件 bbox / center ---
        ctx.mainA_bbox = selected.get(CFG.CLASS_A)
        ctx.mainA_center = DetectionContext._center_of_bbox(ctx.mainA_bbox)
        ctx.mainB_bbox = selected.get(CFG.CLASS_B)
        ctx.mainB_center = DetectionContext._center_of_bbox(ctx.mainB_bbox)
        ctx.mainC_bbox = selected.get(CFG.CLASS_C)
        ctx.mainC_center = DetectionContext._center_of_bbox(ctx.mainC_bbox)

        # --- 螺絲起子 ---
        ctx.screwdriver_bbox = selected.get(CFG.CLASS_SCREWDRIVER)
        ctx.screwdriver_center = DetectionContext._center_of_bbox(ctx.screwdriver_bbox)
        ctx.screwdriver_vertical = DetectionContext._is_vertical(
            ctx.screwdriver_bbox, CFG.SCREWDRIVER_VERTICAL_RATIO
        )

        # --- 手部 ---
        ctx.hand_centers = hand_centers
        ctx.hand_bboxes = hand_bboxes

        # --- ROI flags ---
        assembly = CFG.ASSEMBLY_ROI
        conveyor = CFG.CONVEYOR_ROI

        ctx.A_in_assembly = DetectionContext._point_in_roi(ctx.mainA_center, assembly)
        ctx.B_in_assembly = DetectionContext._point_in_roi(ctx.mainB_center, assembly)
        ctx.C_in_assembly = DetectionContext._point_in_roi(ctx.mainC_center, assembly)
        ctx.screwdriver_in_assembly = DetectionContext._point_in_roi(
            ctx.screwdriver_center, assembly
        )

        # 組裝區內的手數量（用於 ready check）
        ctx.hands_in_assembly_count = sum(
            1 for hc in hand_centers
            if DetectionContext._point_in_roi(hc, assembly)
        )

        # product_in_conveyor: 用原始偵測結果檢查 conveyor（因為 selector 只聽 assembly ROI）
        any_in_assembly = ctx.A_in_assembly or ctx.B_in_assembly or ctx.C_in_assembly
        any_in_conveyor_raw = False
        for cls_name in [CFG.CLASS_A, CFG.CLASS_B, CFG.CLASS_C]:
            for bbox in detections_by_class.get(cls_name, []):
                c = DetectionContext._center_of_bbox(bbox)
                if DetectionContext._point_in_roi(c, conveyor):
                    any_in_conveyor_raw = True
                    break
            if any_in_conveyor_raw:
                break
        ctx.product_in_conveyor = (not any_in_assembly) and any_in_conveyor_raw

        # --- 預計算距離 ---
        # B 到 A、C 到 AB：用中心距離
        # screwdriver 到 target、hand 到 screwdriver：用 bbox 邊緣距離

        # B 到 A（A 可能被遮擋，用最後已知位置 — 中心距離）
        a_ref_center = ctx.mainA_center or self.current_state.mainA.center
        ctx.dist_B_to_A = DetectionContext._dist(ctx.mainB_center, a_ref_center)

        # C 到 AB 核心位置（中心距離）
        ab_pos = self.current_state.AB_pos
        ctx.dist_C_to_AB = DetectionContext._dist(ctx.mainC_center, ab_pos)

        # screwdriver 到目標：依步驟決定 target，用 bbox 邊緣距離（優先）或中心距離
        screw_target_bbox, screw_target_center = self._get_screw_target()
        if screw_target_bbox is not None:
            # 有 bbox：用 point-to-bbox 距離
            ctx.dist_screwdriver_to_target = DetectionContext._point_to_bbox_dist(
                ctx.screwdriver_center, screw_target_bbox
            )
        else:
            # 只有 center：用 point-to-point 距離
            ctx.dist_screwdriver_to_target = DetectionContext._dist(
                ctx.screwdriver_center, screw_target_center
            )

        # hand 到 screwdriver：優先 bbox-to-bbox，否則 point-to-bbox
        ctx.dist_hand_to_screwdriver = self._calc_hand_screwdriver_dist(
            ctx.screwdriver_bbox, ctx.screwdriver_center,
            hand_bboxes, hand_centers,
        )

        return ctx

    @staticmethod
    def _calc_hand_screwdriver_dist(
        screw_bbox: Optional[Tuple[int, int, int, int]],
        screw_center: Optional[Tuple[int, int]],
        hand_bboxes: List[Tuple[int, int, int, int]],
        hand_centers: List[Tuple[int, int]],
    ) -> float:
        """計算手到螺絲起子的最短距離（優先 bbox 邊緣距離）"""
        # bbox-to-bbox：最精確
        if screw_bbox is not None and hand_bboxes:
            return min(
                DetectionContext._bbox_to_bbox_dist(hb, screw_bbox)
                for hb in hand_bboxes
            )
        # point-to-bbox：手中心到螺絲起子 bbox
        if screw_bbox is not None and hand_centers:
            return min(
                DetectionContext._point_to_bbox_dist(hc, screw_bbox)
                for hc in hand_centers
            )
        # fallback: center-to-center
        return DetectionContext._min_dist_to_list(screw_center, hand_centers)

    # ------------------------------------------------------------------

    def _get_screw_target(self) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int]]]:
        """
        根據當前流程步驟決定螺絲起子的比較目標位置。
        
        Returns:
            (target_bbox, target_center)
        """
        state = self.current_state
        if state.C_on_AB_candidate and not state.second_screw_done:
            # Step6: 比較 ABC_pos 或 C 的 bbox
            return (state.mainC.bbox, state.ABC_pos or state.AB_pos)
        if state.AB_candidate and not state.first_screw_done:
            # Step4: 比較 AB_pos 或 B 的 bbox
            return (state.mainB.bbox, state.AB_pos or state.mainB.center)
        return (None, None)

    # ==================================================================
    # 對外查詢
    # ==================================================================

    def get_state_dict(self) -> dict:
        return self.current_state.to_dict()

    def get_history(self) -> List[dict]:
        return list(self.history)

    def get_latest_completed(self) -> Optional[dict]:
        if self.history:
            return self.history[-1]
        return None
