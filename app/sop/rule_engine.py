"""
SOPRuleEngine - SOP 流程規則引擎

吃 DetectionContext，更新 SOPState。
每個 step 拆成獨立 method，方便維護與調整條件。
"""
import logging
from typing import List

from ..config import SOPConfig
from .state import SOPState
from .detection_context import DetectionContext

logger = logging.getLogger(__name__)

CFG = SOPConfig


class SOPRuleEngine:
    """
    SOP 規則引擎。

    使用方式：
        engine = SOPRuleEngine()
        events = engine.update(state, ctx)

    每幀回傳 events list，例如：
        [{"type": "step_changed", "step": 3}, ...]
    """

    def update(self, state: SOPState, ctx: DetectionContext) -> List[dict]:
        """
        根據 DetectionContext 更新 SOPState，回傳產生的事件列表。
        """
        events: List[dict] = []
        old_step = state.current_step

        # 先更新主物件的 seen / stable / missing 狀態
        self._update_object_tracking(state, ctx)

        # 依序檢查各步驟
        # Step1 / Step2 不強制先後
        if not state.has_A:
            self._update_step1(state, ctx)
        if not state.has_B:
            self._update_step2(state, ctx)

        # Step3~7 順序往下
        if state.has_A and state.has_B and not state.AB_confirmed:
            self._update_step3(state, ctx)

        if state.AB_candidate and not state.first_screw_done:
            # Step3 rollback 檢查
            self._check_step3_rollback(state, ctx)
            # Step4
            self._update_step4(state, ctx)

        if state.first_screw_done and not state.C_on_AB_confirmed:
            self._update_step5(state, ctx)

        if state.C_on_AB_candidate and not state.second_screw_done:
            # Step5 rollback 檢查
            self._check_step5_rollback(state, ctx)
            # Step6
            self._update_step6(state, ctx)

        if state.second_screw_done and not state.product_done:
            self._update_step7(state, ctx)

        # 計算 current_step
        state.current_step = self._calc_current_step(state)

        if state.current_step != old_step:
            events.append({
                "type": "step_changed",
                "step": state.current_step,
                "product_id": state.product_id,
            })
            logger.info(f"[SOP] Step {old_step} → {state.current_step} "
                        f"(product={state.product_id})")

        if state.product_done:
            events.append({
                "type": "product_completed",
                **state.summary(),
            })
            logger.info(f"[SOP] 產品完成 product={state.product_id}")

        # 組合 debug_msg
        state.debug_msg = self._build_debug_msg(state, ctx)

        return events

    # ==================================================================
    # 主物件 seen / stable / missing 狀態
    # ==================================================================

    def _update_object_tracking(self, state: SOPState, ctx: DetectionContext):
        """更新主物件的 stable / missing 計數"""
        for obj_state, center_val, in_assembly in [
            (state.mainA, ctx.mainA_center, ctx.A_in_assembly),
            (state.mainB, ctx.mainB_center, ctx.B_in_assembly),
            (state.mainC, ctx.mainC_center, ctx.C_in_assembly),
        ]:
            if center_val is not None:
                obj_state.seen = True
                obj_state.center = center_val
                obj_state.missing_frames = 0
                if in_assembly:
                    obj_state.stable_frames += 1
                else:
                    # 不在 ROI 內，不累計
                    obj_state.stable_frames = 0
                obj_state.last_seen_frame = ctx.frame_idx
            else:
                obj_state.seen = False
                obj_state.missing_frames += 1
                # 不立刻清除 stable_frames，允許遮擋容忍
                # 只有超過 tolerance 才重置
                if obj_state.missing_frames > CFG.MISSING_FRAMES_TOLERANCE:
                    obj_state.stable_frames = 0

            # 確認是否穩定
            if obj_state.stable_frames >= CFG.STABLE_FRAMES_REQUIRED:
                obj_state.confirmed = True

        # 額外把 bbox 也更新
        state.mainA.bbox = ctx.mainA_bbox
        state.mainB.bbox = ctx.mainB_bbox
        state.mainC.bbox = ctx.mainC_bbox

    # ==================================================================
    # Step 1: A 在組裝區穩定存在
    # ==================================================================

    def _update_step1(self, state: SOPState, ctx: DetectionContext):
        if state.mainA.confirmed:
            state.has_A = True
            logger.debug("[SOP] Step1 完成: A 穩定存在於組裝區")

    # ==================================================================
    # Step 2: B 在組裝區穩定存在（與 Step1 不強制順序）
    # ==================================================================

    def _update_step2(self, state: SOPState, ctx: DetectionContext):
        if state.mainB.confirmed:
            state.has_B = True
            logger.debug("[SOP] Step2 完成: B 穩定存在於組裝區")

    # ==================================================================
    # Step 3: B 放到 A 上 → AB_candidate
    # ==================================================================

    def _update_step3(self, state: SOPState, ctx: DetectionContext):
        """
        條件：
          - A 曾在 assembly ROI（has_A=True）
          - B 靠近 A 的最後位置
          - 累計幾幀後成立 AB_candidate
        """
        if state.AB_candidate:
            return  # 已經是 candidate 了

        # A 的最後已知位置
        a_pos = state.mainA.center
        if a_pos is None:
            return

        if ctx.dist_B_to_A >= 0 and ctx.dist_B_to_A <= CFG.DIST_B_TO_A:
            state.step3_candidate_frames += 1
        else:
            # 允許少量中斷，不一次清零
            state.step3_candidate_frames = max(0, state.step3_candidate_frames - 1)

        if state.step3_candidate_frames >= CFG.STABLE_FRAMES_REQUIRED:
            state.AB_candidate = True
            # 記住 AB 的核心位置（用 A 的最後位置）
            state.AB_pos = a_pos
            logger.info("[SOP] Step3: AB_candidate 成立")

    # ==================================================================
    # Step 3 Rollback
    # ==================================================================

    def _check_step3_rollback(self, state: SOPState, ctx: DetectionContext):
        """
        在 first_screw_done=False 期間：
        如果 A 又穩定單獨出現 → rollback 取消 AB_candidate
        """
        if state.first_screw_done:
            return
        if not state.AB_candidate:
            return

        # A 單獨出現 = A seen 且 B 不 seen
        if state.mainA.seen and not state.mainB.seen:
            state.step3_rollback_frames += 1
        else:
            state.step3_rollback_frames = 0

        if state.step3_rollback_frames >= CFG.ROLLBACK_STEP3_FRAMES:
            state.AB_candidate = False
            state.AB_pos = None
            state.step3_candidate_frames = 0
            state.step3_rollback_frames = 0
            logger.info("[SOP] Step3 Rollback: A 又單獨出現，取消 AB_candidate")

    # ==================================================================
    # Step 4: 第一段鎖螺絲
    # ==================================================================

    def _update_step4(self, state: SOPState, ctx: DetectionContext):
        """
        條件：
          - AB_candidate = True
          - screwdriver 靠近 AB_pos 或 B 位置
          - hand 靠近 screwdriver
          - screwdriver 呈垂直
        """
        if state.first_screw_done:
            return
        if not state.AB_candidate:
            return

        screw_close = False
        if ctx.dist_screwdriver_to_target >= 0:
            screw_close = ctx.dist_screwdriver_to_target <= CFG.DIST_SCREWDRIVER_TO_TARGET

        hand_close = False
        if ctx.dist_hand_to_screwdriver >= 0:
            hand_close = ctx.dist_hand_to_screwdriver <= CFG.DIST_HAND_TO_SCREWDRIVER

        if screw_close and hand_close and ctx.screwdriver_vertical:
            state.step4_frames += 1
        else:
            state.step4_frames = max(0, state.step4_frames - 1)

        if state.step4_frames >= CFG.SCREW_FRAMES_REQUIRED:
            state.first_screw_done = True
            state.AB_confirmed = True
            # 更新 AB_pos 到最新位置
            if state.AB_pos is None and state.mainA.center:
                state.AB_pos = state.mainA.center
            logger.info("[SOP] Step4 完成: 第一段鎖螺絲完成，AB_confirmed")

    # ==================================================================
    # Step 5: C 放到 AB 上 → C_on_AB_candidate
    # ==================================================================

    def _update_step5(self, state: SOPState, ctx: DetectionContext):
        """
        條件：
          - first_screw_done = True
          - C 靠近 AB_pos
        """
        if state.C_on_AB_candidate:
            return

        if state.AB_pos is None:
            return

        if ctx.dist_C_to_AB >= 0 and ctx.dist_C_to_AB <= CFG.DIST_C_TO_AB:
            state.step5_candidate_frames += 1
        else:
            state.step5_candidate_frames = max(0, state.step5_candidate_frames - 1)

        if state.step5_candidate_frames >= CFG.STABLE_FRAMES_REQUIRED:
            state.C_on_AB_candidate = True
            state.ABC_pos = state.AB_pos  # 核心位置沿用
            logger.info("[SOP] Step5: C_on_AB_candidate 成立")

    # ==================================================================
    # Step 5 Rollback
    # ==================================================================

    def _check_step5_rollback(self, state: SOPState, ctx: DetectionContext):
        """
        在 second_screw_done=False 期間：
        如果 B 又穩定單獨出現且 C 離開核心區 → rollback
        """
        if state.second_screw_done:
            return
        if not state.C_on_AB_candidate:
            return

        b_alone = state.mainB.seen and not state.mainC.seen
        c_far = ctx.dist_C_to_AB < 0 or ctx.dist_C_to_AB > CFG.DIST_C_TO_AB * 2

        if b_alone and c_far:
            state.step5_rollback_frames += 1
        else:
            state.step5_rollback_frames = 0

        if state.step5_rollback_frames >= CFG.ROLLBACK_STEP5_FRAMES:
            state.C_on_AB_candidate = False
            state.ABC_pos = None
            state.step5_candidate_frames = 0
            state.step5_rollback_frames = 0
            logger.info("[SOP] Step5 Rollback: B 又單獨出現，取消 C_on_AB_candidate")

    # ==================================================================
    # Step 6: 第二段鎖螺絲
    # ==================================================================

    def _update_step6(self, state: SOPState, ctx: DetectionContext):
        """條件類似 Step4，但 target 換成 ABC_pos"""
        if state.second_screw_done:
            return
        if not state.C_on_AB_candidate:
            return

        screw_close = False
        if ctx.dist_screwdriver_to_target >= 0:
            screw_close = ctx.dist_screwdriver_to_target <= CFG.DIST_SCREWDRIVER_TO_TARGET

        hand_close = False
        if ctx.dist_hand_to_screwdriver >= 0:
            hand_close = ctx.dist_hand_to_screwdriver <= CFG.DIST_HAND_TO_SCREWDRIVER

        if screw_close and hand_close and ctx.screwdriver_vertical:
            state.step6_frames += 1
        else:
            state.step6_frames = max(0, state.step6_frames - 1)

        if state.step6_frames >= CFG.SCREW_FRAMES_REQUIRED:
            state.second_screw_done = True
            state.C_on_AB_confirmed = True
            logger.info("[SOP] Step6 完成: 第二段鎖螺絲完成，C_on_AB_confirmed")

    # ==================================================================
    # Step 7: 成品放到輸送帶
    # ==================================================================

    def _update_step7(self, state: SOPState, ctx: DetectionContext):
        """
        當 second_screw_done=True，且主體離開 assembly ROI 並進入 conveyor ROI。
        """
        if state.product_done:
            return

        if ctx.product_in_conveyor:
            state.step7_frames += 1
        else:
            state.step7_frames = max(0, state.step7_frames - 1)

        if state.step7_frames >= CFG.PRODUCT_LEAVE_FRAMES:
            state.product_done = True
            logger.info("[SOP] Step7 完成: 產品已放到輸送帶")

    # ==================================================================
    # 計算 current_step
    # ==================================================================

    @staticmethod
    def _calc_current_step(state: SOPState) -> int:
        if state.product_done:
            return 7
        if state.second_screw_done:
            return 7  # 等待放到輸送帶
        if state.C_on_AB_candidate:
            return 6  # 等待第二段鎖螺絲
        if state.first_screw_done:
            return 5  # 等待放 C
        if state.AB_candidate:
            return 4  # 等待第一段鎖螺絲
        if state.has_A and state.has_B:
            return 3  # 等待 B 放到 A 上
        if state.has_A or state.has_B:
            return max(1 if state.has_A else 0, 2 if state.has_B else 0)
        return 0

    # ==================================================================
    # Debug msg
    # ==================================================================

    @staticmethod
    def _build_debug_msg(state: SOPState, ctx: DetectionContext) -> str:
        parts = []
        parts.append(f"Step={state.current_step}")
        if state.has_A:
            parts.append("A✓")
        if state.has_B:
            parts.append("B✓")
        if state.AB_candidate:
            parts.append("AB?")
        if state.AB_confirmed:
            parts.append("AB✓")
        if state.first_screw_done:
            parts.append("Screw1✓")
        if state.C_on_AB_candidate:
            parts.append("C+AB?")
        if state.C_on_AB_confirmed:
            parts.append("C+AB✓")
        if state.second_screw_done:
            parts.append("Screw2✓")
        if state.product_done:
            parts.append("DONE✓")
        return " | ".join(parts)
