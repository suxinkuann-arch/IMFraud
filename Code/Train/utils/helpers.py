import torch
import gc

def setup_environment(config):
    """设置环境"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

def memory_cleanup():
    """内存清理"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_torch_dtype(dtype_str):
    """获取torch数据类型"""
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    else:
        return torch.float32