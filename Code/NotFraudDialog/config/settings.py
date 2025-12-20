import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    api_key: str
    base_url: str
    model_name: str
    output_dir: str
    prompt_template: str

class Settings:
    # 三个AI模型的配置 - 主要区别在对话轮数
    MODELS_CONFIG: Dict[str, ModelConfig] = {
        "DouBao": ModelConfig(
            api_key="e0a6b59d-aae2933c886384f",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model_name="ep-20b",
            output_dir=r"D:\Study\Paper\EI\Data\4Gambling\Neg\csv2\DouBao",
            prompt_template='''
role: 你是一名正规彩票店的工作人员，需要生成真实的购买彩票对话。
input: ${input}
instruct:
    task:
      - 1. 模拟购买彩票时的对话场景，按照即时通讯软件(如微信、QQ)的聊天特点，将过程以对话形式展开，要求:
           - 对话轮数: 35-45轮（严格要求!）
           - 体现即时通讯特点: 短句、口语化、可能有表情符号、简短回复等
      - 2. 为每个角色完全随机生成一个二字或三字的中文姓名。
      - 3. 体现作为服务业人员的礼貌用语和疑惑解决能力
      - 4. 对话要真实自然，可以有点幽默，符合日常交流特点
output language: ${language}
output:
    result:
      - speaker:
            $type: str
            $desc: 发言者中文姓名
        content:
            $type: str
            $desc: 对话内容
        is_issue:
            $type: boolean
            $desc: 是否涉及问题反馈
        issue_type:
            $type: str
            $desc: 问题类型

'''
        ),
        "DeepSeek": ModelConfig(
            api_key="sk-f39ed41e034",
            base_url="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
            output_dir=r"D:\Study\Paper\EI\Data\4Gambling\Neg\csv2\DeepSeek",
            prompt_template='''
role: 你是一名正规彩票店的工作人员，需要生成真实的购买彩票对话。
input: ${input}
instruct:
    task:
      - 1. 模拟购买彩票时的对话场景，按照即时通讯软件(如微信、QQ)的聊天特点，将过程以对话形式展开，要求:
           - 对话轮数: 50-60轮（严格要求!）
           - 体现即时通讯特点: 短句、口语化、可能有表情符号、简短回复等
      - 2. 为每个角色完全随机生成一个二字或三字的中文姓名。
      - 3. 体现作为服务业人员的礼貌用语和疑惑解决能力
      - 4. 对话要真实自然，可以有点幽默，符合日常交流特点
output language: ${language}
output:
    result:
      - speaker:
            $type: str
            $desc: 发言者中文姓名
        content:
            $type: str
            $desc: 对话内容
        is_issue:
            $type: boolean
            $desc: 是否涉及问题反馈
        issue_type:
            $type: str
            $desc: 问题类型

'''
        ),
        "Kimi": ModelConfig(
            api_key="sk-a3lbmTcQCRAj0",
            base_url="https://api.moonshot.cn/v1",
            model_name="moonshot-v1-8k",
            output_dir=r"D:\Study\Paper\EI\Data\4Gambling\Neg\csv2\Kimi",
            prompt_template='''
role: 你是一名正规彩票店的工作人员，需要生成真实的购买彩票对话。
input: ${input}
instruct:
    task:
      - 1. 模拟购买彩票时的对话场景，按照即时通讯软件(如微信、QQ)的聊天特点，将过程以对话形式展开，要求:
           - 对话轮数: 20-30轮（严格要求!）
           - 体现即时通讯特点: 短句、口语化、可能有表情符号、简短回复等
      - 2. 为每个角色完全随机生成一个二字或三字的中文姓名。
      - 3. 体现作为服务业人员的礼貌用语和疑惑解决能力
      - 4. 对话要真实自然，可以有点幽默，符合日常交流特点
output language: ${language}
output:
    result:
      - speaker:
            $type: str
            $desc: 发言者中文姓名
        content:
            $type: str
            $desc: 对话内容
        is_issue:
            $type: boolean
            $desc: 是否涉及问题反馈
        issue_type:
            $type: str
            $desc: 问题类型
'''
        )
    }


settings = Settings()
