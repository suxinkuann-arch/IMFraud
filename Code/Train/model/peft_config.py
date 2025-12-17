from peft import LoraConfig, TaskType, get_peft_model


def build_peft_model(model, lora_config):
    """构建PEFT模型"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config["target_modules"],
        inference_mode=False,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"]
    )
    peft_model = get_peft_model(model, config)

    # 打印可训练参数
    peft_model.print_trainable_parameters()

    return peft_model