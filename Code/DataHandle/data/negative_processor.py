"""
反例（正常）数据处理器
"""
import os
import pandas as pd
from typing import List, Dict, Any

from config import settings
from .base_processor import BaseDataProcessor
from utils.file_utils import FileUtils


class NegativeDataProcessor(BaseDataProcessor):
    """处理反例（正常）数据"""

    def process_dialogs(self) -> List[Dict[str, Any]]:
        """处理正常对话数据"""
        print("开始处理正常对话（反例）...")
        base_input_dir = settings.BASE_PATHS['negative_csv']
        column_map = settings.COLUMN_NAMES['negative']

        for model_dir in settings.MODELS:
            input_dir = os.path.join(base_input_dir, model_dir)
            files = FileUtils.get_files(input_dir, '.csv')

            if not files:
                print(f"在 {input_dir} 中未找到CSV文件")
                continue

            for file in files:
                file_path = os.path.join(input_dir, file)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')

                    # 检查必要列
                    required_cols = [column_map['group'], column_map['speaker'], column_map['content']]
                    if not all(col in df.columns for col in required_cols):
                        print(f"文件 {file} 缺少必要列，跳过")
                        continue

                    # 分组处理
                    grouped = df.groupby(column_map['group'])

                    for max_len in settings.SPLIT_LENGTHS:
                        for _, group in grouped:
                            dialogs = self._split_dialog(group, max_len, column_map, False)
                            self.processed_data.extend(dialogs)

                    print(f"  已处理: {file}")

                except Exception as e:
                    print(f"  错误处理文件 {file}: {str(e)}")

        print(f"正常对话处理完成，生成 {len(self.processed_data)} 个片段")
        return self.processed_data