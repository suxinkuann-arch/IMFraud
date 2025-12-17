import math

# 模型参数P（单位：十亿）
P_values = {
    "Qwen3-0.6B": 0.6,
    "Qwen3-1.7B": 1.7,
    "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.0
}

# 从表格中提取F1值（每类诈骗的第二列F1值）
# 注意：Fraud4的数据全部为"--"，在计算平均值时跳过
model_f1_scores = {
    "Qwen3-0.6B": [0.9872, 0.9445, 1.0000, None, 0.9836, 0.9913],  # Fraud1-Fraud6
    "Qwen3-1.7B": [0.9873, 0.9691, 0.9927, None, 0.9876, 0.9978],
    "Qwen3-4B": [0.9952, 0.9734, 1.0000, None, 0.9937, 0.9978],
    "Qwen3-8B": [0.9935, 0.9626, 0.9723, None, 0.9877, 0.9889]
}

# 计算Pmin和Pmax
P_min = min(P_values.values())
P_max = max(P_values.values())

print(f"P_min: {P_min}")
print(f"P_max: {P_max}\n")

# 计算每个模型的LightF1
for model_name, f1_scores in model_f1_scores.items():
    # 1. 计算F1_average（跳过None值）
    valid_f1_scores = [score for score in f1_scores if score is not None]
    F1_average = sum(valid_f1_scores) / len(valid_f1_scores)

    # 2. 计算P_norm
    P = P_values[model_name]
    numerator = math.log(1 + P) - math.log(1 + P_min)
    denominator = math.log(1 + P_max) - math.log(1 + P_min)
    P_norm = 1 + (numerator / denominator)

    # 3. 计算LightF1
    LightF1 = F1_average / P_norm

    print(f"模型: {model_name}")
    print(f"  参数数量P: {P} 十亿")
    print(f"  F1值: {valid_f1_scores}")
    print(f"  F1_average: {F1_average:.6f}")
    print(f"  P_norm: {P_norm:.6f}")
    print(f"  LightF1: {LightF1:.6f}")
    print("-" * 50)

# 可选：比较不同模型的LightF1
print("\n模型LightF1排名（从高到低）：")
model_results = {}
for model_name, f1_scores in model_f1_scores.items():
    valid_f1_scores = [score for score in f1_scores if score is not None]
    F1_average = sum(valid_f1_scores) / len(valid_f1_scores)
    P = P_values[model_name]
    numerator = math.log(1 + P) - math.log(1 + P_min)
    denominator = math.log(1 + P_max) - math.log(1 + P_min)
    P_norm = 1 + (numerator / denominator)
    LightF1 = F1_average / P_norm
    model_results[model_name] = LightF1

# 按LightF1值排序
sorted_models = sorted(model_results.items(), key=lambda x: x[1], reverse=True)
for i, (model_name, lightf1) in enumerate(sorted_models, 1):
    print(f"  {i}. {model_name}: {lightf1:.6f}")