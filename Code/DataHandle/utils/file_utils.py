"""
文件操作工具类
"""
import os
import pandas as pd
import json
from typing import List, Optional, Any
from pathlib import Path


class FileUtils:
    """文件操作工具类"""

    @staticmethod
    def get_files(directory: str, extension: str = '.csv') -> List[str]:
        """获取目录下指定扩展名的所有文件"""
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) if f.endswith(extension)]

    @staticmethod
    def read_file(file_path: str, file_type: str = 'csv', **kwargs) -> Any:
        """读取文件内容，支持多种格式"""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            if file_type == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_type == 'jsonl':
                return FileUtils.read_jsonl(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            raise Exception(f"读取文件失败 {file_path}: {str(e)}")

    @staticmethod
    def read_jsonl(file_path: str) -> List[dict]:
        """读取JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    @staticmethod
    def write_jsonl(data: List[dict], file_path: str):
        """写入JSONL文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def filename(path: str, with_extension: bool = False) -> str:
        """提取文件名"""
        if with_extension:
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]