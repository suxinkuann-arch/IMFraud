"""
配置文件 - 集中管理所有参数和路径
"""
import os
from typing import Dict, List

# 基础路径配置
BASE_PATHS: Dict[str, str] = {
    'positive_csv': r"D:\Study\Paper\EI\Data\4Gambling\Pos\csv",
    'negative_csv': r"D:\Study\Paper\EI\Data\4Gambling\Neg\csv",
    'output': r"D:\Study\Paper\EI\Data\4Gambling\Finally"
}

# 使用的模型目录
MODELS: List[str] = ['DeepSeek', 'DouBao', 'Kimi']

# 对话分割长度阈值
SPLIT_LENGTHS: List[int] = [326,639,1000,2000]

# CSV文件列名映射
COLUMN_NAMES: Dict[str, Dict[str, str]] = {
    'positive': {
        'group': 'case',
        'speaker': 'speaker',
        'content': 'content',
        'fraud': 'is_fraud'
    },
    'negative': {
        'group': 'case_id',
        'speaker': 'speaker',
        'content': 'content',
        'turn_number': 'turn_number'
    }
}

# 数据处理参数
MIN_DIALOG_LENGTH: int = 83
ANONYMIZE_SPEAKERS: bool = True
INSTRUCTION: str = "下面是一段对话文本, 请分析对话内容是否有诈骗风险，只以json格式输出你的判断结果(is_fraud: true/false)。"

# 数据集划分比例
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1
RANDOM_STATE: int = 42