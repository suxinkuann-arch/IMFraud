"""
文本处理工具类
"""
import re
from typing import List, Dict, Any


class TextUtils:
    """文本处理工具"""

    @staticmethod
    def total_chars(strs: List[str]) -> int:
        """计算字符串列表的总字符数"""
        return sum(len(s) for s in strs)

    @staticmethod
    def replace_speakers(dialog_text: str) -> str:
        """将对话中的说话人替换为A和B"""
        lines = dialog_text.split('\n')
        speaker_mapping = {}
        processed_lines = []

        for line in lines:
            match = re.match(r'^([^:\n：]+?)[:：]\s*(.*)$', line)
            if match:
                original_speaker = match.group(1).strip()
                content = match.group(2)

                # 清理角色名称
                cleaned_speaker = re.sub(r'[^\w\u4e00-\u9fa5]', '', original_speaker)

                if not cleaned_speaker:
                    processed_lines.append(line)
                    continue

                # 为新说话者分配标识
                if cleaned_speaker not in speaker_mapping:
                    if len(speaker_mapping) == 0:
                        speaker_mapping[cleaned_speaker] = 'A'
                    else:
                        # 查找是否已存在映射
                        for known_speaker, assigned in speaker_mapping.items():
                            if cleaned_speaker == known_speaker:
                                speaker_mapping[cleaned_speaker] = assigned
                                break
                        else:
                            speaker_mapping[cleaned_speaker] = 'B'

                new_speaker = speaker_mapping[cleaned_speaker]
                processed_lines.append(f"{new_speaker}: {content}")
            else:
                processed_lines.append(line)

        return '\n'.join(processed_lines)