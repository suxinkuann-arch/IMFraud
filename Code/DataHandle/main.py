"""
主程序入口 - 组装完整处理流程
"""
from data.positive_processor import PositiveDataProcessor
from data.negative_processor import NegativeDataProcessor
from utils.data_utils import DataUtils
from utils.text_utils import TextUtils
from utils.file_utils import FileUtils
from config import settings


def main():
    """执行完整的数据处理流程"""
    print("=" * 60)
    print("开始执行诈骗数据处理流程")
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

    # 5. 匿名化处理
    if settings.ANONYMIZE_SPEAKERS:
        print("开始匿名化处理...")
        for item in deduplicated_data:
            item['input'] = TextUtils.replace_speakers(item['input'])
        print("匿名化处理完成")

    # 6. 长度过滤
    filtered_data = [item for item in deduplicated_data
                     if len(item['input']) >= settings.MIN_DIALOG_LENGTH]
    print(f"长度过滤: {len(deduplicated_data)}条 → {len(filtered_data)}条")

    # 7. 划分数据集
    train_data, eval_data, test_data = DataUtils.stratified_split_dataset(
        filtered_data,
        settings.TRAIN_RATIO,
        settings.VAL_RATIO
    )

    # 8. 均衡化处理
    balanced_train = DataUtils.balance_dataset(train_data, "训练集")
    balanced_eval = DataUtils.balance_dataset(eval_data, "验证集")
    balanced_test = DataUtils.balance_dataset(test_data, "测试集")

    # 9. 保存结果
    output_dir = settings.BASE_PATHS['output']
    FileUtils.write_jsonl(balanced_train, f"{output_dir}/train.jsonl")
    FileUtils.write_jsonl(balanced_eval, f"{output_dir}/eval.jsonl")
    FileUtils.write_jsonl(balanced_test, f"{output_dir}/test.jsonl")

    # 10. 输出摘要
    print("\n" + "=" * 60)
    print("数据处理完成摘要:")
    print(f"最终训练集: {len(balanced_train)}条")
    print(f"最终验证集: {len(balanced_eval)}条")
    print(f"最终测试集: {len(balanced_test)}条")
    print(f"数据已保存到: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()