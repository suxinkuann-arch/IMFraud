"""主程序入口 - 组装完整处理流程"""
from data.positive_processor import PositiveDataProcessor
from data.negative_processor import NegativeDataProcessor
from utils.data_utils import DataUtils
from utils.text_utils import TextUtils
from utils.file_utils import FileUtils
from config import settings


def main():
    """执行完整的数据处理流程"""
    print("=" * 60)
    print("开始执行冒充客服诈骗数据处理流程")
    print("=" * 60)

    # 1. 初始化处理器
    positive_processor = PositiveDataProcessor()
    negative_processor = NegativeDataProcessor()

    # 2. 处理正例和反例数据
    positive_data = positive_processor.process_dialogs()
    negative_data = negative_processor.process_dialogs()

    # 3. 合并数据
    combined_data = positive_data + negative_data
    print(f"\n数据合并: 正例{len(positive_data)}条 + 反例{len(negative_data)}条 = 总计{len(combined_data)}条")

    # 4. 去重
    deduplicated_data = DataUtils.deduplicate_data(combined_data)
    print(f"去重后数据: {len(deduplicated_data)}条")

    # 5. 匿名化处理（保留此步骤）
    if settings.ANONYMIZE_SPEAKERS:
        print("开始匿名化处理...")
        for item in deduplicated_data:
            item['input'] = TextUtils.replace_speakers(item['input'])
        print("匿名化处理完成")

        # 可选：保存匿名化后的数据
        output_dir = settings.BASE_PATHS['output']
        FileUtils.write_jsonl(deduplicated_data, f"{output_dir}/anonymized_data.jsonl")
        print(f"匿名化数据已保存到: {output_dir}/anonymized_data.jsonl")

    # 6. 在此处停止执行后续流程
    print("\n" + "=" * 60)
    print("数据处理流程已提前结束")
    print(f"已完成步骤: 数据合并 → 去重 → 匿名化处理")
    print(f"生成数据: {len(deduplicated_data)}条")
    print("后续步骤（长度过滤、数据集划分、均衡化等）已跳过")
    print("=" * 60)

    return deduplicated_data  # 返回处理到当前步骤的数据




if __name__ == "__main__":
    # 执行到匿名化后停止
    main()