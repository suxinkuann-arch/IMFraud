import os
import torch
from transformers import AutoTokenizer

from config.config import TrainingConfig
from data.data_loader import load_dataset, view_data_distribution
from data.preprocess import preprocess_datasets
from model.model_loader import load_model, cleanup_memory
from model.peft_config import build_peft_model
from training.trainer_builder import build_trainer
from utils.helpers import setup_environment, memory_cleanup, get_torch_dtype

import warnings


def main():
    warnings.filterwarnings("ignore")
    # 设置配置
    config = TrainingConfig()
    setup_environment(config)

    # 设置分布式训练
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"开始训练，使用设备: {local_rank}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH,
        use_fast=False,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        print("检测到tokenizer缺少pad_token，正在设置...")
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"设置pad_token为eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("添加新的pad_token: [PAD]")

    # 查看数据分布
    if local_rank == 0:
        view_data_distribution(config.TRAIN_DATA_PATH, show_first=True)

    # 加载和预处理数据
    train_ds, eval_ds = load_dataset(config.TRAIN_DATA_PATH, config.EVAL_DATA_PATH)
    train_dataset, eval_dataset = preprocess_datasets(
        train_ds, eval_ds, tokenizer, config.MAX_LENGTH
    )

    if local_rank == 0:
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(eval_dataset)}")
        print(f"Pad token: {tokenizer.pad_token}")
        print(f"Pad token ID: {tokenizer.pad_token_id}")

    # 清理内存
    memory_cleanup()

    # 加载模型
    torch_dtype = get_torch_dtype(config.TORCH_DTYPE)
    model = load_model(config.MODEL_PATH, torch_dtype)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"设置模型config.pad_token_id为: {tokenizer.pad_token_id}")

    if tokenizer.pad_token is not None and tokenizer.pad_token not in tokenizer.get_vocab():
        model.resize_token_embeddings(len(tokenizer))
        print("调整模型词嵌入大小以匹配新的pad_token")

    # 构建LoRA配置
    lora_config = {
        "target_modules": config.LORA_TARGET_MODULES,
        "r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT
    }

    # 构建PEFT模型
    peft_model = build_peft_model(model, lora_config)

    # 构建训练器并开始训练
    trainer = build_trainer(
        peft_model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config.OUTPUT_PATH,
        config
    )

    # 开始训练
    print("开始模型训练...")
    trainer.train()

    # 保存最终模型
    if local_rank == 0:
        trainer.save_model()
        print(f"训练完成，模型已保存到: {config.OUTPUT_PATH}")

    # 最终内存清理
    cleanup_memory()


if __name__ == "__main__":
    main()









#BMB专用train
# import os
# import torch
# from transformers import AutoTokenizer
#
# from config.config import TrainingConfig
# from data.data_loader import load_dataset, view_data_distribution
# from data.preprocess import preprocess_datasets
# from model.model_loader import load_model, cleanup_memory
# from model.peft_config import build_peft_model
# from training.trainer_builder import build_trainer
# from utils.helpers import setup_environment, memory_cleanup, get_torch_dtype
#
# import warnings
#
# # 🔧 底层修复：MiniCPM4 模型兼容性补丁
# import transformers
# from transformers.cache_utils import DynamicCache
#
# # 保存原始的 _reorder_cache 函数
# original_reorder_cache = getattr(transformers, "_reorder_cache", None)
#
#
# def patched_reorder_cache(past_key_values, beam_idx):
#     """修复 MiniCPM4 模型的缓存格式问题"""
#     if isinstance(past_key_values, tuple) and original_reorder_cache is not None:
#         # 将旧格式转换为新格式
#         cache = DynamicCache.from_legacy_cache(past_key_values)
#         return cache
#     return past_key_values
#
#
# # 替换 transformers 的 _reorder_cache 函数
# transformers._reorder_cache = patched_reorder_cache
#
#
# # 修复模型的 forward 方法
# def patch_model_forward(model):
#     """为 MiniCPM4 模型打补丁，修复 past_key_values 格式问题"""
#     if hasattr(model, "forward") and hasattr(model, "config"):
#         original_forward = model.forward
#
#         def new_forward(*args, **kwargs):
#             # 检查并转换 past_key_values 格式
#             if "past_key_values" in kwargs and isinstance(kwargs["past_key_values"], tuple):
#                 try:
#                     kwargs["past_key_values"] = DynamicCache.from_legacy_cache(kwargs["past_key_values"])
#                 except Exception as e:
#                     print(f"[DEBUG] 转换 past_key_values 时出错: {e}")
#                     kwargs["past_key_values"] = None
#
#             # 调用原始 forward
#             outputs = original_forward(*args, **kwargs)
#
#             # 确保输出中的 past_key_values 也是正确格式
#             if hasattr(outputs, "past_key_values") and isinstance(outputs.past_key_values, tuple):
#                 outputs.past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
#
#             return outputs
#
#         model.forward = new_forward
#     return model
#
#
# def main():
#     warnings.filterwarnings("ignore")
#     # 设置配置
#     config = TrainingConfig()
#     setup_environment(config)
#
#     # 设置分布式训练
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#
#     print(f"开始训练，使用设备: {local_rank}")
#
#     # 🔧 关键修复：设置 use_cache=False 以避免与梯度检查点冲突
#     config.USE_CACHE = False  # 如果 config 中没有这个属性，可以在模型加载时设置
#
#     # 加载tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         config.MODEL_PATH,
#         use_fast=False,
#         trust_remote_code=True
#     )
#
#     # 🔧 关键修复：设置padding token
#     if tokenizer.pad_token is None:
#         print("检测到tokenizer缺少pad_token，正在设置...")
#         if tokenizer.eos_token is not None:
#             tokenizer.pad_token = tokenizer.eos_token
#             print(f"设置pad_token为eos_token: {tokenizer.eos_token}")
#         else:
#             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#             print("添加新的pad_token: [PAD]")
#
#     # 查看数据分布
#     if local_rank == 0:
#         view_data_distribution(config.TRAIN_DATA_PATH, show_first=True)
#
#     # 加载和预处理数据
#     train_ds, eval_ds = load_dataset(config.TRAIN_DATA_PATH, config.EVAL_DATA_PATH)
#     train_dataset, eval_dataset = preprocess_datasets(
#         train_ds, eval_ds, tokenizer, config.MAX_LENGTH
#     )
#
#     if local_rank == 0:
#         print(f"训练集大小: {len(train_dataset)}")
#         print(f"验证集大小: {len(eval_dataset)}")
#         print(f"Pad token: {tokenizer.pad_token}")
#         print(f"Pad token ID: {tokenizer.pad_token_id}")
#
#     # 清理内存
#     memory_cleanup()
#
#     # 加载模型
#     torch_dtype = get_torch_dtype(config.TORCH_DTYPE)
#     model = load_model(config.MODEL_PATH, torch_dtype)
#
#     # 🔧 关键修复 1：应用模型补丁
#     print("🔧 应用 MiniCPM4 模型兼容性补丁...")
#     model = patch_model_forward(model)
#
#     # 🔧 关键修复 2：设置 use_cache=False
#     if hasattr(model.config, "use_cache"):
#         print("🔧 设置 use_cache=False 以避免与梯度检查点冲突")
#         model.config.use_cache = False
#
#     # 🔧 关键修复 3：确保模型配置与tokenizer一致
#     if model.config.pad_token_id is None:
#         model.config.pad_token_id = tokenizer.pad_token_id
#         print(f"设置模型config.pad_token_id为: {tokenizer.pad_token_id}")
#
#     # 如果添加了新的pad_token，需要调整模型词嵌入大小
#     if hasattr(tokenizer, 'get_vocab') and tokenizer.pad_token not in tokenizer.get_vocab():
#         model.resize_token_embeddings(len(tokenizer))
#         print("调整模型词嵌入大小以匹配新的pad_token")
#
#     # 清理内存
#     memory_cleanup()
#
#     # 构建LoRA配置
#     lora_config = {
#         "target_modules": config.LORA_TARGET_MODULES,
#         "r": config.LORA_R,
#         "lora_alpha": config.LORA_ALPHA,
#         "lora_dropout": config.LORA_DROPOUT
#     }
#
#     # 构建PEFT模型
#     peft_model = build_peft_model(model, lora_config)
#
#     # 🔧 关键修复 4：在构建 trainer 前再次应用补丁
#     print("🔧 再次应用模型补丁到 PEFT 模型...")
#     if hasattr(peft_model, "base_model"):
#         patch_model_forward(peft_model.base_model)
#     else:
#         patch_model_forward(peft_model)
#
#     # 构建训练器
#     trainer = build_trainer(
#         peft_model,
#         tokenizer,
#         train_dataset,
#         eval_dataset,
#         config.OUTPUT_PATH,
#         config
#     )
#
#     # 🔧 关键修复 5：添加分布式资源清理
#     import atexit
#     import torch.distributed as dist
#
#     def cleanup_distributed():
#         """清理分布式训练资源"""
#         if dist.is_initialized():
#             print("🔧 清理分布式训练资源...")
#             try:
#                 dist.barrier()
#                 dist.destroy_process_group()
#                 print("✅ 分布式资源已清理")
#             except Exception as e:
#                 print(f"⚠️ 清理分布式资源时出错: {e}")
#
#     atexit.register(cleanup_distributed)
#     print("✅ 已注册分布式资源清理函数")
#
#     # 开始训练
#     print("🚀 开始模型训练...")
#     try:
#         trainer.train()
#     except Exception as e:
#         print(f"❌ 训练过程中出错: {e}")
#         # 确保在异常时也清理资源
#         cleanup_distributed()
#         raise
#
#     # 保存最终模型
#     if local_rank == 0:
#         try:
#             trainer.save_model()
#             print(f"✅ 训练完成，模型已保存到: {config.OUTPUT_PATH}")
#         except Exception as e:
#             print(f"⚠️ 保存模型时出错: {e}")
#
#     # 手动调用清理
#     cleanup_distributed()
#
#     # 最终内存清理
#     cleanup_memory()
#     print("✅ 训练过程完全结束")
#
#
# if __name__ == "__main__":
#     main()