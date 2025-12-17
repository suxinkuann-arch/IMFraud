from transformers import TrainingArguments

def build_training_arguments(output_path, config):
    """构建训练参数"""
    return TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        logging_steps=config.LOGGING_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        learning_rate=config.LEARNING_RATE,
        save_on_each_node=True,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,  # 并行训练重要参数
        dataloader_pin_memory=False,  # 减少内存使用
        remove_unused_columns=False,
        report_to=[],  # 不报告到任何平台
        bf16=True,  # 使用bfloat16节省内存
    )