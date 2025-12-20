# import os
# from dataclasses import dataclass
# from typing import Dict, Any
#
#
# @dataclass
# class ModelConfig:
#     """AI模型配置类"""
#     api_key: str
#     base_url: str
#     model_name: str
#     output_dir: str
#
#
# class Settings:
#     """项目配置设置"""
#
#     # 基础路径配置
#     BASE_DATASET_PATH = r"D:\Study\Paper\Data\冒充客服诈骗\A.正例\0.新闻报道"
#     COLUMN_CONTENT = "文章正文内容"
#
#     # AI模型配置
#     MODELS_CONFIG: Dict[str, ModelConfig] = {
#         "DouBao": ModelConfig(
#             api_key="e0a6b886384f",
#             base_url="https://ark.cn-beijing.volces.com/api/v3",
#             model_name="ep-20250603202608-psbpt",
#             output_dir=r"D:\Study\Paper\Data\冒充客服诈骗\A.正例\1.对话csv文件\DouBao"
#         ),
#         "DeepSeek": ModelConfig(
#             api_key="sk-16e3f3340fa2b2",
#             base_url="https://api.deepseek.com/v1",
#             model_name="deepseek-chat",
#             output_dir=r"D:\Study\Paper\Data\冒充客服诈骗\A.正例\1.对话csv文件\DeepSeek"
#         ),
#         "Kimi": ModelConfig(
#             api_key="sk-kYL2gI8qWa0jX6XN7lFffY2L6",
#             base_url="https://api.moonshot.cn/v1",
#             model_name="moonshot-v1-8k",
#             output_dir=r"D:\Study\Paper\Data\冒充客服诈骗\A.正例\1.对话csv文件\Kimi"
#         )
#     }
#
#     # 对话生成参数
#     PROMPT_TEMPLATE = '''

# '''
#
#
# settings = Settings()


from dataclasses import dataclass
#dataclass装饰器用于简化类的定义，自动生成_init_、_repr_等方法,常用于存储数据的类。
from typing import Dict
#Dict是类型注解工具，用于指定变量应为字典类型


@dataclass
class ModelConfig:
    api_key: str                              #api_key
    base_url: str                             #API服务的基础地质
    model_name: str                           #具体模型标识
    output_dir: str                           #结果输出目录
    prompt_template: str                      #提示词目录
#数据类,声明所包含的数据名称和数据类型。



#配置类
class Settings:
    BASE_DATASET_PATH = r"D:\Study\Paper\EI\Data\4Gambling\Pos\News"
    #新闻报道路径
    COLUMN_CONTENT = "文章正文内容"
    #csv新闻报道对应的正文列名

    #MODELS_CONFIG:变量名。Dict[str, ModelConfig]为字典类型，字典键类型为str，值类型为ModelConfig。
    MODELS_CONFIG: Dict[str, ModelConfig] = {


        #DouBao的ModelConfig相关量赋值
        "DouBao": ModelConfig(
            api_key="e0a6b59d-aae2-4283-9022-5933c886384f",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model_name="ep-20250603202608-psbpt",
            output_dir=r"D:\Study\Paper\EI\Data\4Gambling\Pos\csv\DouBao",
            prompt_template='''
role: 你是一个分析诈骗案例并且擅长写作的专家，你的任务是将诈骗的案发过程以口语对话的形式还原。
input: ${input}
instruct:
    task:
      - 1. 分析{input}中描述的案发过程，识别出诈骗者和受害者角色，并为每个角色生成一个中文姓名。
      - 2. 按照即时通讯软件(如微信、QQ)的聊天特点，将案发过程以对话形式展开，要求:
           - 对话轮数: 35-45轮（严格要求!）
           - 体现即时通讯特点: 短句、口语化、可能有表情符号、简短回复等
      - 3. 诈骗者的发言需要逐步建立信任，避免直接暴露欺诈意图。
      - 4. 对每条发言分析是否存在欺诈内容，并标记is_fraud。
output language: ${language}
output:
    result:
      - speaker:
            $type: str
            $desc: 发言者中文姓名
        content:
            $type: str
            $desc: 符合即时通讯特点的对话内容
        is_fraud:
            $type: boolean
            $desc: 是否存在欺诈内容
        reason:
            $type: str
            $desc: 仅当is_fraud=true时给出的理由
'''
        ),


        # DeepSeek的ModelConfig相关量赋值
        "DeepSeek": ModelConfig(
            api_key="sk-f39ed21442744903866119bd5241e034",
            base_url="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
            output_dir=r"D:\Study\Paper\EI\Data\4Gambling\Pos\csv\DeepSeek",
            prompt_template='''
role: 你是一个分析诈骗案例并且擅长写作的专家，你的任务是将诈骗的案发过程以口语对话的形式还原。
input: ${input}
instruct:
    task:
      - 1. 分析{input}中描述的案发过程，识别出诈骗者和受害者角色，并为每个角色生成一个中文姓名。
      - 2. 按照即时通讯软件(如微信、QQ)的聊天特点，将案发过程以对话形式展开，要求:
           - 对话轮数: 50-60轮（严格要求!）
           - 体现即时通讯特点: 短句、口语化、可能有表情符号、简短回复等
      - 3. 诈骗者的发言需要逐步建立信任，避免直接暴露欺诈意图。
      - 4. 对每条发言分析是否存在欺诈内容，并标记is_fraud。
output language: ${language}
output:
    result:
      - speaker:
            $type: str
            $desc: 发言者中文姓名
        content:
            $type: str
            $desc: 符合即时通讯特点的对话内容
        is_fraud:
            $type: boolean
            $desc: 是否存在欺诈内容
        reason:
            $type: str
            $desc: 仅当is_fraud=true时给出的理由
'''
        ),


        # Kimi的ModelConfig相关量赋值
        "Kimi": ModelConfig(
            api_key="sk-a326c5NUVWg3ORriwWQNalNPes64K8Fs6szRvlbmTcQCRAj0",
            base_url="https://api.moonshot.cn/v1",
            model_name="moonshot-v1-8k",
            output_dir=r"D:\Study\Paper\EI\Data\4Gambling\Pos\csv\Kimi",
            prompt_template='''
role: 你是一个分析诈骗案例并且擅长写作的专家，你的任务是将诈骗的案发过程以口语对话的形式还原。
input: ${input}
instruct:
    task:
      - 1. 分析{input}中描述的案发过程，识别出诈骗者和受害者角色，并为每个角色生成一个中文姓名。
      - 2. 按照即时通讯软件(如微信、QQ)的聊天特点，将案发过程以对话形式展开，要求:
           - 对话轮数: 20-30轮（严格要求!）
           - 体现即时通讯特点: 短句、口语化、可能有表情符号、简短回复等
      - 3. 诈骗者的发言需要逐步建立信任，避免直接暴露欺诈意图。
      - 4. 对每条发言分析是否存在欺诈内容，并标记is_fraud。
output language: ${language}
output:
    result:
      - speaker:
            $type: str
            $desc: 发言者中文姓名
        content:
            $type: str
            $desc: 符合即时通讯特点的对话内容
        is_fraud:
            $type: boolean
            $desc: 是否存在欺诈内容
        reason:
            $type: str
            $desc: 仅当is_fraud=true时给出的理由
'''
        )
    }


# 无参构造settings实例,作为模型的配置文件。
settings = Settings()



#Prompt：类yaml结构

#role：指定角色

#input:${input}在提示词中标记出一个空位，这个空位会在程序运行时被具体的值动态填充和替换。{}的作用是：划定变量范围的边界。$的作用是：作为变量名的标识符前缀。

#instruct：指令。task:任务，让大模型按指定步骤执行。

#output language:指定语言。

#output:输出。result：-表示数组,description:说明。
#最后返回一个数组，数组的每一个元素均为speaker+content+is_fraud+reason。一个数组就是一个对话, 一个数组的每一个元素就是对话中的一句话。


