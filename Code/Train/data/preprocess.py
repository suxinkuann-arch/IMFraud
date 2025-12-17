import json
from transformers import AutoTokenizer


def preprocess_function(item, tokenizer, max_length=2048):
    """预处理函数"""
    system_message = "You are a helpful assistant."
    user_message = item['instruction'] + item['input']
    assistant_message = json.dumps({"is_fraud": item["label"]}, ensure_ascii=False)

    instruction = tokenizer(
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False
    )
    response = tokenizer(assistant_message, add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断到最大长度
    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length]
    }


def preprocess_datasets(train_ds, eval_ds, tokenizer, max_length=2048):
    """预处理数据集"""

    def preprocess_fn(examples):
        return preprocess_function(examples, tokenizer, max_length)

    train_dataset = train_ds.map(
        preprocess_fn,
        remove_columns=train_ds.column_names,
        desc="Processing training data"
    )

    eval_dataset = eval_ds.map(
        preprocess_fn,
        remove_columns=eval_ds.column_names,
        desc="Processing evaluation data"
    )

    return train_dataset, eval_dataset