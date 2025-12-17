import torch
from transformers import AutoModelForCausalLM
import gc


def load_model(model_path, torch_dtype=torch.bfloat16):
    """加载模型"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.enable_input_require_grads()  # 开启梯度检查点时需要

    # 清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return model


def cleanup_memory():
    """清理内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()