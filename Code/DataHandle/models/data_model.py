"""
数据模型定义 - 使用dataclass定义数据结构
"""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainingSample:
    """训练样本数据模型"""
    input: str  # 对话内容
    label: bool  # 标签（True=诈骗，False=正常）
    instruction: str  # 指令文本

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "label": self.label,
            "instruction": self.instruction
        }