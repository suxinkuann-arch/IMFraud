import pandas as pd
import os
import threading
from config.settings import settings
from utils.helpers import remove_markdown_boundary

class DialogGenerator:

    def __init__(self, model_name):
        self.model_name = model_name
        self.agent = None
        self.output_dir = ""
        self.lock = threading.Lock()
    #init函数，用于初始化类。也指明了类的属性。


    def initialize_agent(self, agent_factory):
        self.agent = agent_factory.create_agent(self.model_name)
        self.output_dir = agent_factory.get_output_dir(self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)
    #init agent类，用于初始化agent。
    #self.agent通过调用agent_factory的create_agent方法创建agent
    #self.output_dir过调用agent_factory的get_output_dir方法返回模型名称


    def generate_dialog(self, background_text):
        try:
            response = self.agent.load_yaml_prompt(
                yaml=settings.MODELS_CONFIG[self.model_name].prompt_template,
                variables={"input": background_text, "language": 'chinese'}
            ).start()

            if isinstance(response, str):
                import json
                response = json.loads(remove_markdown_boundary(response))

            return response.get('result', [])
        except Exception as e:
            print(f"{self.model_name} 生成错误: {e}")
            return []

    def write_dialog(self, file_name, dialog, case_prefix, case_index, is_first_file=False):
        with self.lock:
            rows = []
            for item in dialog:
                rows.append({
                    'case': f'{case_prefix}_{case_index}',
                    'speaker': item.get('speaker', ''),
                    'content': item.get('content', ''),
                    'is_fraud': item.get('is_fraud', ''),
                    'reason': item.get('reason', ''),
                    'model': self.model_name
                })

            df_new = pd.DataFrame(rows)
            header = not os.path.exists(file_name) or is_first_file
            df_new.to_csv(file_name, mode='a', header=header, index=False)
            print(f"{self.model_name} - 案例{case_index}: {len(dialog)}轮")







# 对象传递：AgentFactory实例是在main_with_background.py中创建，然后作为参数传递给DialogGenerator的
#
# 2.
# 不需要导入：dialog_generator.py不需要导入AgentFactory类，因为它接收的是对象实例，不是类定义
#
# 3.
# 动态语言特性：Python是动态语言，只要运行时对象有相应的方法就能正常工作
#
# 4.
# 设计模式：这是一种叫做"依赖注入"的设计模式，让代码更灵活、更容易测试
#
# 简单来说：dialog_generator.py中的agent_factory不是通过import来的，而是别人创建好后"送"给它的。
#
# 这样解释清楚了吗？如果还有不明白的地方，请告诉我具体是哪个环节还不理解。