"""
数据操作工具类
"""
import random
from random import sample
from collections import OrderedDict
from typing import List, Dict, Any, Tuple


class DataUtils:
    """数据操作工具"""

    @staticmethod
    def deduplicate_data(data_list: List[Dict[str, Any]], key_field: str = "input") -> List[Dict[str, Any]]:
        """基于指定字段对数据列表进行去重"""
        seen = OrderedDict()
        duplicate_count = 0

        for item in data_list:
            if key_field not in item:
                continue

            key_value = item[key_field]
            if not isinstance(key_value, str):
                import json
                key_value = json.dumps(key_value, sort_keys=True, ensure_ascii=False)

            if key_value in seen:
                duplicate_count += 1
            seen[key_value] = item

        deduplicated_data = list(seen.values())
        print(f"去重完成: 原始{len(data_list)}条 → 去重后{len(deduplicated_data)}条, 重复{duplicate_count}条")
        return deduplicated_data

    @staticmethod
    def stratified_split_dataset(dataset: List[Dict[str, Any]],
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 random_state: int = 42) -> Tuple[List, List, List]:
        """使用分层抽样划分数据集)"""
        random.seed(random_state)

        # 按标签分离数据
        true_data = [d for d in dataset if d['label'] == True]
        false_data = [d for d in dataset if d['label'] == False]

        # 打乱数据
        random.shuffle(true_data)
        random.shuffle(false_data)

        # 计算划分点
        true_train_size = int(len(true_data) * train_ratio)
        true_val_size = int(len(true_data) * val_ratio)

        false_train_size = int(len(false_data) * train_ratio)
        false_val_size = int(len(false_data) * val_ratio)

        # 划分数据
        true_train = true_data[:true_train_size]
        true_val = true_data[true_train_size:true_train_size + true_val_size]
        true_test = true_data[true_train_size + true_val_size:]

        false_train = false_data[:false_train_size]
        false_val = false_data[false_train_size:false_train_size + false_val_size]
        false_test = false_data[false_train_size + false_val_size:]

        # 合并并打乱
        train_data = true_train + false_train
        val_data = true_val + false_val
        test_data = true_test + false_test

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        return train_data, val_data, test_data

    @staticmethod
    def balance_dataset(dataset: List[Dict[str, Any]], dataset_name: str = "") -> List[Dict[str, Any]]:
        """对数据集进行均衡化处理（下采样）"""
        true_data = [d for d in dataset if d['label'] == True]
        false_data = [d for d in dataset if d['label'] == False]

        min_count = min(len(true_data), len(false_data))
        if min_count == 0:
            print(f"警告: {dataset_name}中某个类别的样本数为0，无法均衡化")
            return dataset

        # 下采样
        if len(true_data) > min_count:
            true_data = sample(true_data, min_count)
        if len(false_data) > min_count:
            false_data = sample(false_data, min_count)

        balanced_data = true_data + false_data
        random.shuffle(balanced_data)

        print(f"{dataset_name}均衡化: {len(dataset)}条 → {len(balanced_data)}条")
        return balanced_data