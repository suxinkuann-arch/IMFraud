import os
import time
from agents.agent_factory import AgentFactory
from core.file_processor import FileProcessor
from utils.helpers import get_files
from config.settings import settings


def main():
    print("开始生成对话...")
    start_time = time.time()

    models = ["DouBao", "DeepSeek", "Kimi"]
    agent_factory = AgentFactory()
    processor = FileProcessor(models)
    processor.initialize_generators(agent_factory)

    csv_files = get_files(settings.BASE_DATASET_PATH, '.csv')
    if not csv_files:
        print("未找到CSV文件")
        return

    for i, csv_file in enumerate(csv_files, 1):
        input_path = os.path.join(settings.BASE_DATASET_PATH, csv_file)
        print(f"处理文件 {i}/{len(csv_files)}: {csv_file}")
        processor.process_single_file(input_path)

    print(f"完成! 耗时: {time.time() - start_time:.1f}秒")


if __name__ == "__main__":
    main()