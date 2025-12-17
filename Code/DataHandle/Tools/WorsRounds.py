import os
import json


def analyze_jsonl_files(directory):
    total_samples = 0
    total_chars = 0
    total_turns = 0
    all_chars = []
    all_turns = []

    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        data = json.loads(line)
                        input_text = data.get('input', '')

                        # 统计字数和对话轮数
                        char_count = len(input_text)
                        turn_count = input_text.count('\n') + 1

                        total_chars += char_count
                        total_turns += turn_count
                        all_chars.append(char_count)
                        all_turns.append(turn_count)
                        total_samples += 1

    # 计算结果
    avg_chars = total_chars / total_samples if total_samples > 0 else 0
    avg_turns = total_turns / total_samples if total_samples > 0 else 0
    min_chars = min(all_chars) if all_chars else 0
    max_chars = max(all_chars) if all_chars else 0
    min_turns = min(all_turns) if all_turns else 0
    max_turns = max(all_turns) if all_turns else 0

    return {
        '样本总数': total_samples,
        '平均字数': round(avg_chars, 2),
        '平均对话轮数': round(avg_turns, 2),
        '最小字数': min_chars,
        '最大字数': max_chars,
        '最小对话轮数': min_turns,
        '最大对话轮数': max_turns
    }


# 使用示例
directory_path = r'D:\Study\Paper\EI\Data\4Gambling\Finally'  # 替换为你的目录路径
results = analyze_jsonl_files(directory_path)

print("统计结果:")
for key, value in results.items():
    print(f"{key}: {value}")