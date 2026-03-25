"""
SOPState - SOP 流程狀態資料結構
儲存當前產品的所有流程狀態
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ObjectState:
    """單一主物件的追蹤狀態"""
    bbox: Optional[Tuple[int, int, int, int]] = None   # (x1, y1, x2, y2)
    center: Optional[Tuple[int, int]] = None            # (cx, cy)
    seen: bool = False          # 這一幀是否有看到
    confirmed: bool = False     # 是否已穩定確認存在於 assembly ROI
    stable_frames: int = 0      # 連續在 assembly ROI 中偵測到的幀數
    missing_frames: int = 0     # 連續未偵測到的幀數
    last_seen_frame: int = -1   # 最後一次被偵測到的 frame_idx

    def reset(self):
        self.bbox = None
        self.center = None
        self.seen = False
        self.confirmed = False
        self.stable_frames = 0
        self.missing_frames = 0
        self.last_seen_frame = -1


class SOPState:
    """
    一件產品的完整 SOP 流程狀態。

    流程步驟：
        Step 1: A 在組裝區穩定存在
        Step 2: B 在組裝區穩定存在（與 Step1 不強制順序）
        Step 3: B 放到 A 上 → AB_candidate → AB_confirmed（可 rollback）
        Step 4: 第一段鎖螺絲 → first_screw_done
        Step 5: C 放到 AB 上 → C_on_AB_candidate → C_on_AB_confirmed（可 rollback）
        Step 6: 第二段鎖螺絲 → second_screw_done
        Step 7: 成品放到輸送帶 → product_done
    """

    def __init__(self):
        self.product_id: str = uuid.uuid4().hex[:8]
        self.created_at: float = time.time()

        # 就緒狀態（手 + 螺絲起子在組裝區）
        self.ready: bool = False
        self.ready_frames: int = 0
        self.ready_check_done: bool = False  # 只判斷一次上工，完成後不再重複檢查

        # 當前步驟 (-1=等待上工, 0=已上工但未開始, 1~7=流程步驟)
        self.current_step: int = -1

        # --- 主物件追蹤 ---
        self.mainA = ObjectState()
        self.mainB = ObjectState()
        self.mainC = ObjectState()

        # --- 流程 flags ---
        self.has_A: bool = False          # Step1 完成
        self.has_B: bool = False          # Step2 完成

        self.AB_candidate: bool = False   # B 靠近 A → 可能疊上去了
        self.AB_confirmed: bool = False   # 第一段鎖螺絲後確認
        self.AB_pos: Optional[Tuple[int, int]] = None  # AB 的核心位置

        self.first_screw_done: bool = False  # Step4 完成

        self.C_on_AB_candidate: bool = False
        self.C_on_AB_confirmed: bool = False
        self.ABC_pos: Optional[Tuple[int, int]] = None  # ABC 的核心位置

        self.second_screw_done: bool = False  # Step6 完成
        self.product_done: bool = False        # Step7 完成

        # --- counters / debounce ---
        self.step3_candidate_frames: int = 0
        self.step4_frames: int = 0
        self.step5_candidate_frames: int = 0
        self.step6_frames: int = 0
        self.step7_frames: int = 0

        # rollback counters
        self.step3_rollback_frames: int = 0
        self.step5_rollback_frames: int = 0

        # --- debug ---
        self.debug_msg: str = ""

    def reset(self):
        """重置為全新產品狀態"""
        self.__init__()

    def to_dict(self) -> dict:
        """轉換為可序列化的 dict（供 WebSocket / API 使用）"""
        return {
            "enabled": True,  # 標記 worker 正在運行
            "product_id": self.product_id,
            "ready": self.ready,
            "current_step": self.current_step,
            "has_A": self.has_A,
            "has_B": self.has_B,
            "AB_candidate": self.AB_candidate,
            "AB_confirmed": self.AB_confirmed,
            "first_screw_done": self.first_screw_done,
            "C_on_AB_candidate": self.C_on_AB_candidate,
            "C_on_AB_confirmed": self.C_on_AB_confirmed,
            "second_screw_done": self.second_screw_done,
            "product_done": self.product_done,
            "debug_msg": self.debug_msg,
        }

    def summary(self) -> dict:
        """產品完成後的摘要"""
        return {
            "product_id": self.product_id,
            "success": self.product_done,
            "created_at": self.created_at,
            "finished_at": time.time(),
            "steps": {
                "step1_has_A": self.has_A,
                "step2_has_B": self.has_B,
                "step3_AB_confirmed": self.AB_confirmed,
                "step4_first_screw": self.first_screw_done,
                "step5_C_on_AB_confirmed": self.C_on_AB_confirmed,
                "step6_second_screw": self.second_screw_done,
                "step7_product_done": self.product_done,
            },
        }
