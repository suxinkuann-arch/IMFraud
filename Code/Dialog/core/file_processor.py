import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.dialog_generator import DialogGenerator
from utils.helpers import get_files, filename
from config.settings import settings

class FileProcessor:
    def __init__(self, models):
        self.models = models
        self.generators = {}

    def initialize_generators(self, agent_factory):
        for model in self.models:
            generator = DialogGenerator(model)
            generator.initialize_agent(agent_factory)
            self.generators[model] = generator

    def process_single_file(self, input_path):
        try:
            dataset = pd.read_csv(input_path)
            #dataset = pd.read_csv(input_path, encoding='gbk')
            case_prefix = filename(input_path)

            # 为每个模型创建输出文件路径
            output_files = {}
            for model in self.models:
                output_dir = self.generators[model].output_dir
                output_files[model] = f"{output_dir}/{case_prefix}.csv"

            # 并行处理
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures = []

                for i, row in dataset.iterrows():
                    background_text = row[settings.COLUMN_CONTENT]

                    for model in self.models:
                        future = executor.submit(
                            self._process_case, model, background_text,
                            output_files[model], case_prefix, i, (i == 0)
                        )
                        futures.append(future)

                # 简单等待完成
                for future in as_completed(futures):
                    future.result()

        except Exception as e:
            print(f"处理文件错误: {e}")

    def _process_case(self, model, background_text, output_file, case_prefix, case_index, is_first_case):
        generator = self.generators[model]
        dialog = generator.generate_dialog(background_text)

        if dialog:
            generator.write_dialog(output_file, dialog, case_prefix, case_index, is_first_case)

        return len(dialog)