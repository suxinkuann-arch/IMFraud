#!/usr/bin/env python3
"""
客服对话生成主程序
三个AI模型并行生成正常客服对话
"""

import time
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dialogue_generator import MultiModelDialogueGenerator


def main():
    """主函数"""
    print("=" * 50)
    #print("     投资对话生成系统（多模型并行生成）")
    print("     贷款对话生成系统（多模型并行生成）")
    print("=" * 50)

    start_time = time.time()

    try:
        # 初始化多模型生成器
        multi_generator = MultiModelDialogueGenerator()

        # 生成对话
        multi_generator.generate_all_dialogues(num_cases=8)

        # 计算耗时
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n生成完成! 总耗时: {total_time:.2f}秒")
        print("输出文件位置:")
        for model in multi_generator.models:
            output_dir = multi_generator.generators[model].output_dir
            print(f"  {model}: {output_dir}/loan_dialogues.csv")

    except Exception as e:
        print(f"程序执行出错: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()