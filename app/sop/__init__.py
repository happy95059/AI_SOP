"""
SOP 組裝流程偵測模組

提供：
- SOPState: 流程狀態資料結構
- DetectionContext: 每幀偵測上下文
- MainObjectSelector: 主物件選擇器
- SOPRuleEngine: 流程規則引擎
- SOPSystem: 總控 / 流程管理器
"""

from .state import SOPState
from .detection_context import DetectionContext
from .selector import MainObjectSelector
from .rule_engine import SOPRuleEngine
from .system import SOPSystem

__all__ = [
    "SOPState",
    "DetectionContext",
    "MainObjectSelector",
    "SOPRuleEngine",
    "SOPSystem",
]
