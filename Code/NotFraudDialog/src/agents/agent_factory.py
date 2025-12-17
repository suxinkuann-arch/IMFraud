import Agently
from config.settings import settings


class AgentFactory:
    """AI代理工厂类"""

    @staticmethod
    def create_agent(model_name: str):
        """创建指定模型的AI代理"""
        if model_name not in settings.MODELS_CONFIG:
            raise ValueError(f"不支持的模型: {model_name}")

        config = settings.MODELS_CONFIG[model_name]

        agent = (
            Agently.create_agent()
            .set_settings("current_model", "OAIClient")
            .set_settings("model.OAIClient.auth", {"api_key": config.api_key})
            .set_settings("model.OAIClient.url", config.base_url)
            .set_settings("model.OAIClient.options", {"model": config.model_name})
        )

        return agent

    @staticmethod
    def get_output_dir(model_name: str) -> str:
        """获取指定模型的输出目录"""
        return settings.MODELS_CONFIG[model_name].output_dir

    @staticmethod
    def get_prompt_template(model_name: str) -> str:
        """获取指定模型的提示词模板"""
        return settings.MODELS_CONFIG[model_name].prompt_template