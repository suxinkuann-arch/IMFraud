"""
基础数据处理器 - 抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

from config.settings import COLUMN_NAMES, INSTRUCTION
from utils.file_utils import FileUtils
from utils.text_utils import TextUtils


class BaseDataProcessor(ABC):
    """数据处理器基类"""

    def __init__(self):
        self.processed_data = []

    def to_train_data(self, dialog: List[str], label: bool) -> Dict[str, Any]:
        """转换为训练数据格式"""
        content = "\n".join(dialog)
        return {
            "input": content,
            "label": label,
            "instruction": INSTRUCTION
        }

    @abstractmethod
    def process_dialogs(self) -> List[Dict[str, Any]]:
        """处理对话数据（子类必须实现）"""
        pass

    def _split_dialog(self, dialog: pd.DataFrame, max_length: int,
                      column_map: Dict[str, str], is_positive: bool = True) -> List[Dict[str, Any]]:
        """通用的对话分割逻辑"""
        small_dialogs = []
        current_segment = []
        current_label = False

        # 确保对话按顺序处理
        if 'turn_number' in dialog.columns:
            dialog = dialog.sort_values('turn_number')

        for _, item in dialog.iterrows():
            statement = f"{item[column_map['speaker']]}: {item[column_map['content']]}"

            if max_length <= 0 or TextUtils.total_chars(current_segment) + len(statement) < max_length:
                current_segment.append(statement)
            else:
                if len(current_segment) >= 2:
                    small_dialogs.append(self.to_train_data(current_segment, current_label))
                current_segment = [statement]
                current_label = False

            # 更新标签（仅正例需要）
            if is_positive and 'fraud' in column_map:
                is_fraud = item[column_map['fraud']]
                if is_fraud == True:
                    current_label = True

        # 处理最后一段
        if len(current_segment) >= 2:
            small_dialogs.append(self.to_train_data(current_segment, current_label))

        return small_dialogs