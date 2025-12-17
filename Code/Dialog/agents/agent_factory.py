import Agently
#Agently框架
from config.settings import settings
#config.settings表示从config.settings包中引入settings(Settings类的实例)


class AgentFactory:
#定义AgentFactory类，用于产生agent。
    @staticmethod
    #静态方法。
    def create_agent(model_name):
        config = settings.MODELS_CONFIG[model_name]
        #设置配置文件config
        return (Agently.create_agent()
                .set_settings("current_model", "OAIClient")
                #指定智能体使用的模型客户端类型为兼容OpenAI API的客户端OAIClient。
                .set_settings("model.OAIClient.auth", {"api_key": config.api_key})
                #{"api_key": config.api_key}字典格式。
                .set_settings("model.OAIClient.url", config.base_url)
                .set_settings("model.OAIClient.options", {"model": config.model_name}))

    @staticmethod
    def get_output_dir(model_name):
        return settings.MODELS_CONFIG[model_name].output_dir
    #返回模型名称，settings实例下的MODELS_CONFIG的[model_name]项下的output_dir