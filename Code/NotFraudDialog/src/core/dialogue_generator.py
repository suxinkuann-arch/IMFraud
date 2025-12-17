import pandas as pd
import os
import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import settings
from src.agents.agent_factory import AgentFactory


class DialogueGenerator:
    """客服对话生成器核心类"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.agent = None
        self.output_dir = ""
        self.prompt_template = ""
        self.lock = threading.Lock()

    def initialize(self):
        """初始化生成器"""
        self.agent = AgentFactory.create_agent(self.model_name)
        self.output_dir = AgentFactory.get_output_dir(self.model_name)
        self.prompt_template = AgentFactory.get_prompt_template(self.model_name)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"{self.model_name} 初始化完成")

    def remove_markdown(self, text: str) -> str:
        """移除Markdown标记"""
        return re.sub(r'^```json\s*(.*?)\s*```$', r'\1', text.strip(), flags=re.DOTALL)

    def generate_dialogue(self, scenario_description: str = "正常的彩票对话") -> list:
        """
        生成客服对话

        Args:
            scenario_description: 场景描述

        Returns:
            生成的对话列表
        """
        try:
            # 构建完整的提示词
            full_prompt = f"生成一个{scenario_description}的彩票对话场景。"

            response = self.agent.load_yaml_prompt(
                yaml=self.prompt_template,
                variables={
                    "input": full_prompt,
                    "language": 'chinese',
                }
            ).start()

            # 处理响应
            if isinstance(response, str):
                response = self.remove_markdown(response)
                response_data = json.loads(response)
            else:
                response_data = response

            dialogue = response_data.get('result', [])
            return dialogue

        except Exception as e:
            print(f"{self.model_name} 生成对话时出错: {e}")
            return []

    def write_dialogue(self, dialogue: list, case_id: int, scenario: str = "彩票对话"):
        """将对话写入CSV文件"""
        if not dialogue:
            return

        output_file = f"{self.output_dir}/Lottery_dialogues.csv"

        with self.lock:
            rows = []
            for i, turn in enumerate(dialogue):
                row_data = {
                    'case_id': f'{self.model_name}_case_{case_id}',
                    'turn_number': i + 1,
                    'speaker': turn.get('speaker', ''),
                    'content': turn.get('content', ''),
                    'is_issue': turn.get('is_issue', ''),
                    'scenario': scenario,
                    'model': self.model_name
                }

                # 添加模型特有字段
                if self.model_name == "DouBao" and 'issue_type' in turn:
                    row_data['issue_type'] = turn.get('issue_type', '')
                elif self.model_name == "DeepSeek" and 'resolution_status' in turn:
                    row_data['resolution_status'] = turn.get('resolution_status', '')
                elif self.model_name == "Kimi" and 'response_time' in turn:
                    row_data['response_time'] = turn.get('response_time', '')

                rows.append(row_data)

            df = pd.DataFrame(rows)
            header = not os.path.exists(output_file)

            df.to_csv(output_file, mode='a', header=header, index=False)
            print(f"{self.model_name} - 案例{case_id}: {len(dialogue)}轮对话")


class MultiModelDialogueGenerator:
    """多模型对话生成管理器"""

    def __init__(self, models: list = None):
        self.models = models or ["DouBao", "DeepSeek", "Kimi"]
        self.generators = {}

        # 初始化所有生成器
        for model in self.models:
            generator = DialogueGenerator(model)
            generator.initialize()
            self.generators[model] = generator

    def generate_all_dialogues(self, num_cases: int = 5):
        """使用所有模型生成对话，各模型完全独立运行"""
        print(f"开始生成{num_cases}个彩票对话案例...")

        scenarios = [
            "篮球体彩",
            "足球体彩",
            "刮刮乐",
            "福利彩票",
            "体育彩票",
            "双色球"
        ]

        # 为每个模型创建独立的线程执行所有案例生成
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            # 提交每个模型的独立生成任务
            future_to_model = {}
            for model in self.models:
                future = executor.submit(
                    self._generate_all_cases_for_model,
                    model, num_cases, scenarios
                )
                future_to_model[future] = model

            print("所有模型生成任务已提交，开始并行生成...")

            # 等待所有模型完成（可选）
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    print(f"---------！！！{model} 生成完成，共生成{result}个案例")
                except Exception as e:
                    print(f"{model} 生成过程中出错: {e}")

    def _generate_all_cases_for_model(self, model: str, num_cases: int, scenarios: list):
        """为单个模型生成所有案例"""
        generator = self.generators[model]  # 这里现在获取的是真正的generator对象
        case_count = 0

        for case_id in range(1, num_cases + 1):
            scenario = scenarios[(case_id - 1) % len(scenarios)]
            try:
                dialogue = generator.generate_dialogue(scenario)
                if dialogue:
                    generator.write_dialogue(dialogue, case_id, scenario)
                    case_count += 1
                else:
                    print(f"{model} - 案例{case_id} 生成失败")
            except Exception as e:
                print(f"{model} - 案例{case_id} 生成时出错: {e}")
                continue

        return case_count