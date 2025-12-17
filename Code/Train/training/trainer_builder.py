from transformers import DataCollatorForSeq2Seq, Trainer, EarlyStoppingCallback
from .trainer_config import build_training_arguments


def build_trainer(model, tokenizer, train_dataset, eval_dataset, output_path, config):
    """构建Trainer"""
    training_args = build_training_arguments(output_path, config)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8  # 优化内存使用
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )