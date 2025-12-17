import json
import pandas as pd
from datasets import Dataset


def load_jsonl(path, encoding='utf-8'):
    """加载JSONL文件"""
    try:
        with open(path, 'r', encoding=encoding) as f:
            return pd.DataFrame([json.loads(line.strip()) for line in f])
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        return pd.DataFrame()


def view_data_distribution(data_path, show_first=False):
    """查看数据分布"""
    df = load_jsonl(data_path, encoding='utf-8')
    print(f"total_count:{df.shape[0]}, true_count: {df['label'].sum()}, false_count: {(df['label'] == False).sum()}")
    if show_first:
        print(json.dumps(df.iloc[0].to_dict(), indent=4, ensure_ascii=False))


def load_dataset(train_path, eval_path):
    """加载训练和验证数据集"""
    train_df = load_jsonl(train_path)
    train_ds = Dataset.from_pandas(train_df)

    eval_df = load_jsonl(eval_path)
    eval_ds = Dataset.from_pandas(eval_df)

    return train_ds, eval_ds