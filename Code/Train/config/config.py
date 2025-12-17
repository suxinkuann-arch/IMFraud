
class TrainingConfig:
    # 数据路径
    TRAIN_DATA_PATH = "/home/sxk1/EI/Data/6Loan/Finally/train.jsonl"
    EVAL_DATA_PATH = "/home/sxk1/EI/Data/6Loan/Finally/eval.jsonl"
    MODEL_PATH = '/home/sxk1/EI/model/ERNIE/PaddlePaddle/ERNIE-4.5-0.3B-PT'
    OUTPUT_PATH = '/home/sxk1/EI/result/CheckPoint/Fraud6/Hunyuan'


    # 训练参数
    MAX_LENGTH = 2048
    #PER_DEVICE_TRAIN_BATCH_SIZE = 4
    #GRADIENT_ACCUMULATION_STEPS = 4
    #若OOM。则用下面这组参数
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 1e-4
    NUM_TRAIN_EPOCHS = 3
    LOGGING_STEPS = 10
    EVAL_STEPS = 10
    SAVE_STEPS = 10
    EARLY_STOPPING_PATIENCE = 3

    # LoRA配置
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # 系统配置
    CUDA_VISIBLE_DEVICES = "0,1,2,3,4"  # 5张显卡
    # CUDA_VISIBLE_DEVICES = "0,1,2,3,4"
    TORCH_DTYPE = "bfloat16"