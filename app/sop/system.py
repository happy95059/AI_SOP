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
        ctx = self._build_context(selected, hand_centers or [])

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

        # --- ROI flags ---
        assembly = CFG.ASSEMBLY_ROI
        conveyor = CFG.CONVEYOR_ROI

        ctx.A_in_assembly = DetectionContext._point_in_roi(ctx.mainA_center, assembly)
        ctx.B_in_assembly = DetectionContext._point_in_roi(ctx.mainB_center, assembly)
        ctx.C_in_assembly = DetectionContext._point_in_roi(ctx.mainC_center, assembly)
        ctx.screwdriver_in_assembly = DetectionContext._point_in_roi(
            ctx.screwdriver_center, assembly
        )

        # product_in_conveyor: 沒有任何主物件在 assembly 但有東西進 conveyor
        # 這裡用一個簡化判斷：assembly 內沒有 A/B/C 且 conveyor 內有偵測
        any_in_assembly = ctx.A_in_assembly or ctx.B_in_assembly or ctx.C_in_assembly
        # 用最後已知核心位置判斷是否進了 conveyor
        core_pos = self.current_state.ABC_pos or self.current_state.AB_pos
        core_in_conveyor = DetectionContext._point_in_roi(core_pos, conveyor) if core_pos else False
        # 也可以檢查是否有任一物件中心在 conveyor
        any_in_conveyor = (
            DetectionContext._point_in_roi(ctx.mainA_center, conveyor)
            or DetectionContext._point_in_roi(ctx.mainB_center, conveyor)
            or DetectionContext._point_in_roi(ctx.mainC_center, conveyor)
        )
        ctx.product_in_conveyor = (not any_in_assembly) and (core_in_conveyor or any_in_conveyor)

        # --- 預計算距離 ---
        # B 到 A（用 A 的最後已知位置）
        a_ref = ctx.mainA_center or self.current_state.mainA.center
        ctx.dist_B_to_A = DetectionContext._dist(ctx.mainB_center, a_ref)

        # C 到 AB 核心位置
        ab_pos = self.current_state.AB_pos
        ctx.dist_C_to_AB = DetectionContext._dist(ctx.mainC_center, ab_pos)

        # screwdriver 到目標：依步驟決定 target
        screw_target = self._get_screw_target()
        ctx.dist_screwdriver_to_target = DetectionContext._dist(
            ctx.screwdriver_center, screw_target
        )

        # hand 到 screwdriver
        ctx.dist_hand_to_screwdriver = DetectionContext._min_dist_to_list(
            ctx.screwdriver_center, hand_centers
        )

        return ctx

    # ------------------------------------------------------------------

    def _get_screw_target(self) -> Optional[Tuple[int, int]]:
        """根據當前流程步驟決定螺絲起子的比較目標位置"""
        state = self.current_state
        if state.C_on_AB_candidate and not state.second_screw_done:
            # Step6: 比較 ABC_pos 或 C 位置
            return state.ABC_pos or state.AB_pos
        if state.AB_candidate and not state.first_screw_done:
            # Step4: 比較 AB_pos 或 B 位置
            return state.AB_pos or state.mainB.center
        return None

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
